import torch

class MLP(torch.nn.Module):
    def __init__(self, window_size: int):
        super(MLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(window_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)