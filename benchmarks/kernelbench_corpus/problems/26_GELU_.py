"""
Small-shape stand-in for KernelBench/KernelBench/level1/26_GELU_.py.
torch.nn.functional.gelu(x) with no `approximate=` kwarg is the EXACT
erf-based formula, not the tanh/sigmoid approximation -- confirmed by
reading the reference file directly, matters for the mutant below.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)


batch_size = 64
dim = 128


def get_inputs():
    x = torch.randn(batch_size, dim) * 4.0
    return [x]


def get_init_inputs():
    return []
