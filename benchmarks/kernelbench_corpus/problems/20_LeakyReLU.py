"""Small-shape stand-in for KernelBench/KernelBench/level1/20_LeakyReLU.py."""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(Model, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)


batch_size = 64
dim = 128


def get_inputs():
    x = torch.randn(batch_size, dim) * 4.0
    return [x]


def get_init_inputs():
    return []
