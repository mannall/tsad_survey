import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

from pathlib import Path

from .donut import ReconstructDataset


class OmniAnomalyModel(nn.Module):
    def __init__(self, device):
        super(OmniAnomalyModel, self).__init__()
        self.name = 'OmniAnomaly'
        self.device = device
        self.lr = 0.002
        self.beta = 0.01

        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(1, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, 2*self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x, hidden = None):
        bs = x.shape[0]
        win = x.shape[1]

        hidden = torch.rand(2, bs, self.n_hidden).to(self.device) if hidden is not None else hidden

        out, hidden = self.lstm(x.view(-1, bs, 1), hidden)

        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        x = mu + eps*std

        x = self.decoder(x)
        return x.reshape(bs, win), mu.reshape(bs, win*self.n_latent), logvar.reshape(bs, win*self.n_latent), hidden


class OmniAnomalyAD():
    def __init__(self,
                 window_size = 16,
                 batch_size = 32,
                 num_epochs = 20,
                 lr = 7e-4,
                 ):
        super().__init__()

        self.device = torch.device("cpu")

        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = num_epochs

        self.model = OmniAnomalyModel(device=self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.criterion = nn.MSELoss(reduction = 'none')


    def train_model(self, train_ts: np.ndarray):
        train_dl = DataLoader(
            dataset=ReconstructDataset(train_ts, self.window_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        mses, klds = [], []
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            for idx, (d, _) in enumerate(train_dl):        
                d = d.to(self.device)
                # print('d: ', d.shape)

                y_pred, mu, logvar, hidden = self.model(d, hidden if idx else None)
                d = d.view(-1, self.window_size)
                MSE = torch.mean(self.criterion(y_pred, d), axis=-1)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                loss = torch.mean(MSE + self.model.beta * KLD)

                mses.append(torch.mean(MSE).item())
                klds.append(self.model.beta * torch.mean(KLD).item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
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


    def score(self, test_ts: np.ndarray):
        test_dl = DataLoader(
            dataset=ReconstructDataset(test_ts, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        self.model.eval()
        scores = []
        y_preds = []
        with torch.no_grad():
            for idx, (d, _) in enumerate(test_dl):
                d = d.to(self.device)

                y_pred, _, _, hidden = self.model(d, hidden if idx else None)
                y_preds.append(y_pred)
                d = d.view(-1, self.window_size)

                loss = torch.mean(self.criterion(y_pred, d), axis=-1)
                scores.append(loss.cpu())
        
        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()

        return scores