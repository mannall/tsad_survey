import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import torch
from torch import nn, optim

from pathlib import Path

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Sequence, Union, Callable

class ReconstructDataset(Dataset):
    def __init__(self, X, window_size):
        super(ReconstructDataset, self).__init__()
        # self.window_size = window_size
        
        self.samples = torch.from_numpy(sliding_window_view(X, window_size)).unsqueeze(-1).float()
        self.targets = self.samples.clone()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]


def sample_normal(mu: torch.Tensor, std_or_log_var: torch.Tensor, log_var: bool = False, num_samples: int = 1):
    # ln(σ) = 0.5 * ln(σ^2) -> σ = e^(0.5 * ln(σ^2))
    if log_var:
        sigma = std_or_log_var.mul(0.5).exp_()
    else:
        sigma = std_or_log_var

    if num_samples == 1:
        eps = torch.randn_like(mu)  # also copies device from mu
    else:
        eps = torch.rand((num_samples,) + mu.shape, dtype=mu.dtype, device=mu.device)
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
    # z = μ + σ * ϵ, with ϵ ~ N(0,I)
    return eps.mul(sigma).add_(mu)


def normal_standard_normal_kl(mean: torch.Tensor, std_or_log_var: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        kl_loss = torch.sum(1 + std_or_log_var - mean.pow(2) - std_or_log_var.exp(), dim=-1)
    else:
        kl_loss = torch.sum(1 + torch.log(std_or_log_var.pow(2)) - mean.pow(2) - std_or_log_var.pow(2), dim=-1)
    return -0.5 * kl_loss
    

def normal_normal_kl(mean_1: torch.Tensor, std_or_log_var_1: torch.Tensor, mean_2: torch.Tensor,
                     std_or_log_var_2: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        return 0.5 * torch.sum(std_or_log_var_2 - std_or_log_var_1 + (torch.exp(std_or_log_var_1)
                               + (mean_1 - mean_2)**2) / torch.exp(std_or_log_var_2) - 1, dim=-1)

    return torch.sum(torch.log(std_or_log_var_2) - torch.log(std_or_log_var_1) \
                     + 0.5 * (std_or_log_var_1**2 + (mean_1 - mean_2)**2) / std_or_log_var_2**2 - 0.5, dim=-1)


class DonutModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, mask_prob) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.mask_prob = mask_prob
        
        encoder = VaeEncoder(input_dim, hidden_dim, latent_dim)
        decoder = VaeEncoder(latent_dim, hidden_dim, input_dim)
        
        self.vae = VAE(encoder=encoder, decoder=decoder, logvar_out=False)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # x: (B, T, D)
        x = inputs
        B, T, D = x.shape

        if self.training:
            # Randomly mask some inputs
            mask = torch.empty_like(x)
            mask.bernoulli_(1 - self.mask_prob)
            x = x * mask
        else:
            mask = None

        # Run the VAE
        x = x.view(x.shape[0], -1)  
        mean_z, std_z, mean_x, std_x, sample_z = self.vae(x, return_latent_sample=True)

        # Reshape the outputs
        mean_x = mean_x.view(B, T, D)
        std_x = std_x.view(B, T, D)
        return mean_z, std_z, mean_x, std_x, sample_z, mask


class VAELoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', logvar_out: bool = True):
        super(VAELoss, self).__init__(size_average, reduce, reduction)
        self.logvar_out = logvar_out

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        z_mean, z_std_or_log_var, x_dec_mean, x_dec_std = predictions[:4]
        if len(predictions) > 4:
            z_prior_mean, z_prior_std_or_logvar = predictions[4:]
        else:
            z_prior_mean, z_prior_std_or_logvar = None, None

        y, = targets

        nll_gauss = F.gaussian_nll_loss(x_dec_mean, y, x_dec_std.pow(2), reduction='none').sum(-1)

        # get KL loss
        if z_prior_mean is None and z_prior_std_or_logvar is None:
            # If a prior is not given, we assume standard normal
            kl_loss = normal_standard_normal_kl(z_mean, z_std_or_log_var, log_var=self.logvar_out)
        else:
            if z_prior_mean is None:
                z_prior_mean = torch.tensor(0, dtype=z_mean.dtype, device=z_mean.device)
            if z_prior_std_or_logvar is None:
                value = 0 if self.logvar_out else 1
                z_prior_std_or_logvar = torch.tensor(value, dtype=z_std_or_log_var.dtype, device=z_std_or_log_var.device)

            kl_loss = normal_normal_kl(z_mean, z_std_or_log_var, z_prior_mean, z_prior_std_or_logvar,
                                       log_var=self.logvar_out)

        # Combine
        final_loss = nll_gauss + kl_loss
        return torch.mean(final_loss)


class MaskedVAELoss(VAELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MaskedVAELoss, self).__init__(size_average, reduce, reduction, logvar_out=False)

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        mean_z, std_z, mean_x, std_x, sample_z, mask = predictions
        actual_x, = targets

        if mask is None:
            mean_z = mean_z.unsqueeze(1)
            std_z = std_z.unsqueeze(1)
            return super(MaskedVAELoss, self).forward((mean_z, std_z, mean_x, std_x), (actual_x,), *args, **kwargs)

        # If the loss is masked, one of the terms in the kl loss is weighted, so we can't compute it exactly
        # anymore and have to use a MC approximation like for the output likelihood
        nll_output = torch.sum(mask * F.gaussian_nll_loss(mean_x, actual_x, std_x**2, reduction='none'), dim=-1)

        # This is p(z), i.e., the prior likelihood of Z. The paper assumes p(z) = N(z| 0, I), we drop constants
        beta = torch.mean(mask, dim=(1, 2)).unsqueeze(-1)
        nll_prior = beta * 0.5 * torch.sum(sample_z * sample_z, dim=-1, keepdim=True)

        nll_approx = torch.sum(F.gaussian_nll_loss(mean_z, sample_z, std_z**2, reduction='none'), dim=-1, keepdim=True)

        final_loss = nll_output + nll_prior - nll_approx

        return torch.mean(final_loss)


class MLP(torch.nn.Module):
    def __init__(self, input_features: int, hidden_layers: Union[int, Sequence[int]], output_features: int,
                 activation: Callable = torch.nn.Identity(), activation_after_last_layer: bool = False):
        super(MLP, self).__init__()

        self.activation = activation
        self.activation_after_last_layer = activation_after_last_layer

        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        layers = [input_features] + list(hidden_layers) + [output_features]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(inp, out) for inp, out in zip(layers[:-1], layers[1:])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers[:-1]:
            out = layer(out)
            out = self.activation(out)

        out = self.layers[-1](out)
        if self.activation_after_last_layer:
            out = self.activation(out)

        return out

class VaeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(VaeEncoder, self).__init__()
        
        self.latent_dim = latent_dim

        self.mlp = MLP(input_dim, hidden_dim, 2*latent_dim, activation=torch.nn.ReLU(), activation_after_last_layer=False)
        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        mlp_out = self.mlp(x)

        mean, std = mlp_out.tensor_split(2, dim=-1)
        std = self.softplus(std)

        return mean, std
    

class VAE(torch.nn.Module):
    """
    VAE Implementation that supports normal distribution with diagonal cov matrix in the latent space
    and the output
    """

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, logvar_out: bool = True):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.log_var = logvar_out

    def forward(self, x: torch.Tensor, return_latent_sample: bool = False, num_samples: int = 1,
                force_sample: bool = False) -> Tuple[torch.Tensor, ...]:
        z_mu, z_std_or_log_var = self.encoder(x)

        if self.training or num_samples > 1 or force_sample:
            z_sample = sample_normal(z_mu, z_std_or_log_var, log_var=self.log_var, num_samples=num_samples)
        else:
            z_sample = z_mu

        x_dec_mean, x_dec_std = self.decoder(z_sample)

        if not return_latent_sample:
            return z_mu, z_std_or_log_var, x_dec_mean, x_dec_std

        return z_mu, z_std_or_log_var, x_dec_mean, x_dec_std, z_sample


class DonutAD():
    def __init__(self,
                 window_size=64,
                 batch_size=32,
                 grad_clip=10.0,
                 num_epochs=10,
                 mc_samples=1024,
                 hidden_dim=128,
                 latent_dim=32,
                 inject_ratio=0.01,
                 learning_rate=3e-4,
                 l2_coff=1e-3):
        super(DonutAD, self).__init__()

        self.device = torch.device("cpu")

        self.window_size = window_size

        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.num_epochs = num_epochs
        self.mc_samples = mc_samples
        
        self.model = DonutModel(input_dim=self.window_size, hidden_dim=hidden_dim, latent_dim=latent_dim, mask_prob=inject_ratio).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=l2_coff)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.75)
        self.vaeloss = MaskedVAELoss()
        
        
    def train_model(self, train_ts):
        train_dl = DataLoader(
            dataset=ReconstructDataset(train_ts, self.window_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.model.train(mode=True)        
        for epoch in range(1, self.num_epochs + 1):
            for x, target in train_dl:
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                output = self.model(x)
                loss = self.vaeloss(output, (target,))
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                
                # avg_loss += loss.cpu().item()
            self.scheduler.step()


    def save(self, model_path: Path):
        torch.save({
            "parameters" : self.__dict__,
            "state_dict": self.model.state_dict(),
        }, model_path)


    def load(self, model_path: Path):
        checkpoint = torch.load(model_path)
        parameters = checkpoint['parameters']
        for key, value in parameters.items():
            setattr(self, key, value)
        self.model.load_state_dict(checkpoint['state_dict'])


    def score(self, test_ts):
        test_dl = DataLoader(
            dataset=ReconstructDataset(test_ts, self.window_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        self.model.eval()
        scores = []
        with torch.no_grad():
            for x, _ in test_dl:
                x = x.to(self.device)
                x_vae = x.view(x.shape[0], -1)
                B, T, D = x.shape

                res = self.model.vae(x_vae, return_latent_sample=False, num_samples=self.mc_samples)
                z_mu, z_std, x_dec_mean, x_dec_std = res

                x_dec_mean = x_dec_mean.view(self.mc_samples, B, T, D)
                x_dec_std = x_dec_std.view(self.mc_samples, B, T, D)                
                nll_output = torch.sum(F.gaussian_nll_loss(x_dec_mean[:, :, -1, :], x[:, -1, :].unsqueeze(0),
                                                   x_dec_std[:, :, -1, :]**2, reduction='none'), dim=(0, 2))
                nll_output /= self.mc_samples

                scores.append(nll_output.cpu())

        scores = torch.cat(scores, dim=0).numpy()
        assert scores.ndim == 1
        return np.r_[scores[:self.window_size-1], scores]
    