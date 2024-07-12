import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torch.nn import Module, Conv1d, MaxPool1d, Linear, Dropout
from torch.utils.data import Dataset, DataLoader

from typing import Iterator, Optional, List, Callable, Tuple

class TimeSeries(Dataset):
    def __init__(self, X, window_length: int, prediction_length: int, output_dims: Optional[List[int]] = None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.output_dims = output_dims or list(range(X.shape[1]))
        self.X = torch.from_numpy(X).float()
        self.window_length = window_length
        self.prediction_length = prediction_length

    def __len__(self):
        return self.X.shape[0] - (self.window_length - 1) - self.prediction_length

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx = index+self.window_length
        x = self.X[index:end_idx].reshape(len(self.output_dims), -1)
        y = self.X[end_idx:end_idx+self.prediction_length, self.output_dims]
        return x, y


class DeepAnTCNN(Module):
    def __init__(self, window_size: int, filter1_size: int, filter2_size: int, kernel_size: int, pool_size: int, stride: int):
        super(DeepAnTCNN, self).__init__()

        self.conv1 = Conv1d(in_channels=1, out_channels=filter1_size, kernel_size=kernel_size, stride=stride, padding = 0)
        self.conv2 = Conv1d(in_channels=filter1_size, out_channels=filter2_size, kernel_size=kernel_size, stride=stride, padding = 0)
        self.maxpool = MaxPool1d(pool_size)
        self.dropout = Dropout(0.25)

        self.dim1 = int(0.5*(0.5*(window_size - 1) - 1)) * filter2_size
        self.lin1 = Linear(self.dim1, 1)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))

        x = x.view(-1, self.dim1)
        x = self.dropout(x)
        x = self.lin1(x)
        return x.view(-1, 1, 1)
    

class DeepAnTCNNAD():
    def __init__(self, 
                 window_size: int = 32, 
                 filter1_size: int = 128, 
                 filter2_size: int = 32, 
                 kernel_size: int = 2, 
                 pool_size: int = 2, 
                 stride: int = 1, 
                 learning_rate: float = 3e-4, 
                 batch_size: int = 32, 
                 num_epochs: int = 15
        ):
        self.model = DeepAnTCNN(window_size, filter1_size, filter2_size, kernel_size, pool_size, stride)
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train_model(self, train_ts: np.ndarray):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        train_dl = DataLoader(
            TimeSeries(train_ts, window_length=self.window_size, prediction_length=1), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.model.train()
        for epoch in range(self.num_epochs):
            train_losses = []
            for X, y in train_dl:
                optimizer.zero_grad()
                output = self.model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())


    def score(self, test_ts: Dataset):
        self.model.eval()

        test_dl = DataLoader(
            TimeSeries(test_ts, window_length=self.window_size, prediction_length=1), 
            batch_size=self.batch_size, 
            shuffle=False
        )
        scores = []
        for x, y in test_dl:
            y_hat = self.model(x).detach()
            scores.append((y - y_hat)**2)

        scores = torch.cat(scores, dim=0).squeeze().numpy()
        return np.r_[scores[:self.window_size], scores]


    def save(self, path):
        torch.save(self.model.state_dict(), path)


    def load(self, path):
        self.model.load_state_dict(torch.load(path))