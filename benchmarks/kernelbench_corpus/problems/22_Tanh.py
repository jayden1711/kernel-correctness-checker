"""Small-shape stand-in for KernelBench/KernelBench/level1/22_Tanh.py."""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


batch_size = 64
dim = 128


def get_inputs():
    x = torch.randn(batch_size, dim) * 3.0
    return [x]


def get_init_inputs():
    return []
