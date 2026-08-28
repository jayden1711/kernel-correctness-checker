"""Small-shape stand-in for KernelBench/KernelBench/level1/29_Softplus.py."""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x)


batch_size = 64
dim = 128


def get_inputs():
    x = torch.randn(batch_size, dim) * 4.0
    return [x]


def get_init_inputs():
    return []
