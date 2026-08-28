"""
Small-shape stand-in for KernelBench/KernelBench/level1/23_Softmax.py.

Same Model contract (dim=1 row-wise softmax), but KernelBench's own
default shape (batch_size=4096, dim=393216 -- ~1.6B elements) is sized
for real performance benchmarking on datacenter GPUs, not a fast
correctness smoke corpus. n_cols=512 here is deliberately > 256 (the
CUDA candidates' BLOCK_SIZE) so a first-tile-only bug actually has a
second tile to leave unprocessed.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)


batch_size = 64
dim = 512


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
