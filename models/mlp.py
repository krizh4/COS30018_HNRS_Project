"""A plain fully connected network (multi-layer perceptron) for MNIST"""
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),              # 1x28x28 picture -> list of 784 numbers
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),     # one score per digit 0-9
        )

    def forward(self, x):
        return self.layers(x)