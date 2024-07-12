import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import numpy as np

from pathlib import Path

class TimeSeries(Dataset):
    def __init__(self, X, window_length: int):
        self.X = torch.from_numpy(X).float()
        self.window_length = window_length

    def __len__(self):
        return self.X.shape[0] - (self.window_length - 1) - 1

    def __getitem__(self, index) -> torch.Tensor:
        end_idx = index+self.window_length
        x = self.X[index:end_idx]
        y = self.X[end_idx:end_idx+1]
        return x, y


class BaseForecastML(nn.Module):
    def __init__(self, window_size: int, learning_rate: float = 3e-4, num_epochs: int = 15, batch_size: int = 16):
        super(BaseForecastML, self).__init__()
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def train_model(self, train_ts: np.ndarray) -> nn.Module:
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        train_dl = DataLoader(
            TimeSeries(train_ts, self.window_size), 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True
        )
        self.train()
        for epoch in range(self.num_epochs):
            for x, y in train_dl:
                y_hat = self(x)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def save(self, model_path: Path):
        torch.save({
            "parameters" : self.__dict__,
            "state_dict": self.state_dict(),
        }, model_path)

    def load(self, model_path: Path):
        checkpoint = torch.load(model_path)
        parameters = checkpoint['parameters']
        for key, value in parameters.items():
            setattr(self, key, value)
        self.load_state_dict(checkpoint['state_dict'])

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        test_dl = DataLoader(
            TimeSeries(test_ts, self.window_size),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        self.eval()
        scores = []
        for x, y in test_dl:
            y_hat = self(x)
            error = (y_hat - y)**2
            scores += error.squeeze().detach().tolist()
        scores = np.array(scores)
        # could use more advanced schemes here(?)
        return np.r_[scores[:self.window_size], scores]


class MLPAD(BaseForecastML):
    def __init__(self, window_size: int, hidden_size: int = 128, **kwargs):
        super(MLPAD, self).__init__(window_size, **kwargs)
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.window_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LSTMAD(BaseForecastML):
    def __init__(self, window_size: int, hidden_size: int, num_layers: int, **kwargs):
        super(LSTMAD, self).__init__(window_size, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstms = nn.LSTM(input_size=1, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.dense = nn.Linear(in_features=self.window_size * self.hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hidden = self.lstms(x.unsqueeze(-1))
        x = self.dense(x.reshape(-1, self.window_size * self.hidden_size))
        return x
    

class TransformerAD(BaseForecastML):
    def __init__(self, window_size: int, hidden_size: int, hidden_feedforward: int, num_heads: int, num_layers: int, **kwargs):
        super(TransformerAD, self).__init__(window_size, **kwargs)
        self.hidden_size = hidden_size
        self.hidden_feedforward = hidden_feedforward
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.ReLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.hidden_size, self.num_heads, self.hidden_feedforward, batch_first = True), self.num_layers
        )
        self.predictor = nn.Linear(self.hidden_size*self.window_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x.unsqueeze(-1))
        x = self.transformer(x)
        return self.predictor(x.reshape(x.shape[0], -1))