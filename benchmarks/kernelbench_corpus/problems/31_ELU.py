"""Small-shape stand-in for KernelBench/KernelBench/level1/31_ELU.py."""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x, alpha=self.alpha)


batch_size = 64
dim = 128


def get_inputs():
    x = torch.randn(batch_size, dim) * 4.0
    return [x]


def get_init_inputs():
    return [1.0]
