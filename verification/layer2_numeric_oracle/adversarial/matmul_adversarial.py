"""
Adversarial inputs for matmul  returns (A, B) pairs.

Targets each cheating kernel:
  partial_k_reduct.py     only accumulates first half of K dimension
  skip_boundary_tiles.py  missing output mask, corrupts non-aligned shapes
  swapped_strides.py      A indexed with B's strides and vice versa
  wrong_dtype.py          accumulates in fp16 instead of fp32
"""

import torch


def k_dim_accumulation(x: torch.Tensor, weight: torch.Tensor) -> tuple:
    """
    A=ones (MxK), B=ones (KxN) -> output must be all-K.
    partial_k_reduct.py only accumulates first K//2 steps -> output is K//2.
    Use large K (>> BLOCK_K=32) to make the difference unmistakable.
    """
    M, K = x.shape
    _, N = weight.shape
    K_large = max(K, 256)
    A = torch.ones(M, K_large, device=x.device, dtype=x.dtype)
    B = torch.ones(K_large, N, device=x.device, dtype=x.dtype)
    return A, B


def non_aligned_shape(x: torch.Tensor, weight: torch.Tensor) -> tuple:
    """
    M, N not multiples of BLOCK_M/BLOCK_N=32.
    skip_boundary_tiles.py omits the output mask  stores corrupt the last tile.
    """
    A = torch.randn(33, 33, device=x.device, dtype=x.dtype)
    B = torch.randn(33, 33, device=x.device, dtype=x.dtype)
    return A, B


def swapped_strides_detector(x: torch.Tensor, weight: torch.Tensor) -> tuple:
    """
    Non-square A and B so that swapping strides produces a clearly wrong result.
    swapped_strides.py uses B's strides for A and A's strides for B.
    With square matrices this is harder to detect; rectangular makes it obvious.

    A: (M, K) with M != K != N so all three strides are different.
    """
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, device=x.device, dtype=x.dtype)
    B = torch.randn(K, N, device=x.device, dtype=x.dtype)
    return A, B


def extreme_dynamic_range(x: torch.Tensor, weight: torch.Tensor) -> tuple:
    """
    Large values expose fp16 accumulation overflow in wrong_dtype.py.
    Values of ~300 cause fp16 overflow (max ~65504) after K=256 accumulations.
    """
    M, K = x.shape
    _, N = weight.shape
    A = torch.randn(M, K, device=x.device, dtype=x.dtype) * 1e2
    B = torch.randn(K, N, device=x.device, dtype=x.dtype) * 1e2
    return A, B


def identity_weight(x: torch.Tensor, weight: torch.Tensor) -> tuple:
    """
    B = identity matrix -> output must equal A exactly.
    Any accumulation or stride bug is immediately detectable.
    """
    M, K = x.shape
    _, N = weight.shape
    A = torch.randn_like(x)
    B = torch.eye(K, N, device=x.device, dtype=x.dtype)
    return A, B


def non_power_of_two(x: torch.Tensor, weight: torch.Tensor) -> tuple:
    """Non-power-of-two dims  exposes tile boundary issues."""
    A = torch.randn(333, 257, device=x.device, dtype=x.dtype)
    B = torch.randn(257, 129, device=x.device, dtype=x.dtype)
    return A, B


def get_adversarial_inputs(x: torch.Tensor, weight: torch.Tensor) -> list:
    return [
        ("k_dim_accumulation",    k_dim_accumulation(x, weight)),
        ("non_aligned_shape",     non_aligned_shape(x, weight)),
        ("swapped_strides",       swapped_strides_detector(x, weight)),
        ("extreme_dynamic_range", extreme_dynamic_range(x, weight)),
        ("identity_weight",       identity_weight(x, weight)),
        ("non_power_of_two",      non_power_of_two(x, weight)),
    ]