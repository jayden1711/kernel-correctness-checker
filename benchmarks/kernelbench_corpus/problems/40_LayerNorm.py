"""
Small-shape stand-in for KernelBench/KernelBench/level1/40_LayerNorm.py.
KernelBench's own default normalizes over 3 joint dims (features x
dim1 x dim2, ~4.2M elements/batch-item) -- sized for perf benchmarking,
not a correctness smoke corpus. normalized_shape=(dim,) here normalizes
over the last axis only, matching the row-wise convention this
project's Triton layernorm work already uses.

nn.LayerNorm(normalized_shape) defaults to weight=ones, bias=zeros,
eps=1e-5 -- the candidates hardcode that same default rather than
threading gamma/beta through as extra forward() args, since
get_init_inputs() (matching real KernelBench convention) only supplies
the shape, not parameter values.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


batch_size = 64
dim = 128


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return [(dim,)]
