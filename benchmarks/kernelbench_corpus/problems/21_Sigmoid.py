"""
Small-shape stand-in for KernelBench/KernelBench/level1/21_Sigmoid.py.
Base scale (+-4ish) deliberately does NOT overflow fp32 exp() -- see
sigmoid_mutant_unstable_exp.py, whose bug only diverges once exp(x)
actually overflows (~x>88), which the checker's own weight_magnitude
adversarial probe (1e4 scale) reaches but this base input does not.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


batch_size = 64
dim = 128


def get_inputs():
    x = torch.randn(batch_size, dim) * 4.0
    return [x]


def get_init_inputs():
    return []
