"""
Small-shape stand-in for
KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py.
KernelBench's own default (N=4096) is sized for perf benchmarking;
N=64 here is plenty to make a K/2-only accumulation bug unmistakable
(all-ones inputs -> correct output is all-64, buggy is all-32) while
staying fast on a naive, unoptimized CUDA kernel.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


N = 64


def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]


def get_init_inputs():
    return []
