"""
Small-shape stand-in for KernelBench/KernelBench/level1/25_Swish.py.
Values scaled to +-4ish (not KernelBench's default [0,1) rand) so a
piecewise-linear sigmoid approximation actually diverges from true
sigmoid somewhere in range -- near x=0 the two are nearly
indistinguishable.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


batch_size = 64
dim = 128


def get_inputs():
    x = torch.randn(batch_size, dim) * 4.0
    return [x]


def get_init_inputs():
    return []
