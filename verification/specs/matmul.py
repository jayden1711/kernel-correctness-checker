"""KernelSpec for matmul — f(A, B) -> Tensor, A:(M,K), B:(K,N)."""

from dataclasses import dataclass, field
from typing import List, Tuple, Callable
import torch

from verification.specs.base_spec import MatmulKernelSpec
from verification.layer3_properties.matmul_properties import (
    check_output_shape,
    check_distributivity,
    check_scalar_associativity,
    check_precision_coercion,
)


# Adversarial input generators -- inlined verbatim from the former
# verification/layer2_numeric_oracle/adversarial/matmul_adversarial.py
# (logic unchanged, only relocated; parameters renamed x/weight -> A/B
# for clarity, matching how they're actually used -- these are the
# captured A, B tensors, not literally "x" and "weight"). Return (A, B)
# pairs.

def _k_dim_accumulation(A: torch.Tensor, B: torch.Tensor) -> tuple:
    """
    A=ones (MxK), B=ones (KxN) -> output must be all-K.
    partial_k_reduct.py only accumulates first K//2 steps -> output is K//2.
    Use large K (>> BLOCK_K=32) to make the difference unmistakable.
    """
    M, K = A.shape
    _, N = B.shape
    K_large = max(K, 256)
    A_out = torch.ones(M, K_large, device=A.device, dtype=A.dtype)
    B_out = torch.ones(K_large, N, device=A.device, dtype=A.dtype)
    return A_out, B_out


def _non_aligned_shape(A: torch.Tensor, B: torch.Tensor) -> tuple:
    """
    M, N not multiples of BLOCK_M/BLOCK_N=32.
    skip_boundary_tiles.py omits the output mask -- stores corrupt the last tile.
    """
    A_out = torch.randn(33, 33, device=A.device, dtype=A.dtype)
    B_out = torch.randn(33, 33, device=A.device, dtype=A.dtype)
    return A_out, B_out


def _swapped_strides_detector(A: torch.Tensor, B: torch.Tensor) -> tuple:
    """
    Non-square A and B so that swapping strides produces a clearly wrong result.
    swapped_strides.py uses B's strides for A and A's strides for B.
    With square matrices this is harder to detect; rectangular makes it obvious.

    A: (M, K) with M != K != N so all three strides are different.
    """
    M, K, N = 64, 128, 32
    A_out = torch.randn(M, K, device=A.device, dtype=A.dtype)
    B_out = torch.randn(K, N, device=A.device, dtype=A.dtype)
    return A_out, B_out


def _extreme_dynamic_range(A: torch.Tensor, B: torch.Tensor) -> tuple:
    """
    Large values expose fp16 accumulation overflow in wrong_dtype.py.
    Values of ~300 cause fp16 overflow (max ~65504) after K=256 accumulations.
    """
    M, K = A.shape
    _, N = B.shape
    A_out = torch.randn(M, K, device=A.device, dtype=A.dtype) * 1e2
    B_out = torch.randn(K, N, device=A.device, dtype=A.dtype) * 1e2
    return A_out, B_out


def _identity_weight(A: torch.Tensor, B: torch.Tensor) -> tuple:
    """
    B = identity matrix -> output must equal A exactly.
    Any accumulation or stride bug is immediately detectable.
    """
    M, K = A.shape
    _, N = B.shape
    A_out = torch.randn_like(A)
    B_out = torch.eye(K, N, device=A.device, dtype=A.dtype)
    return A_out, B_out


def _non_power_of_two(A: torch.Tensor, B: torch.Tensor) -> tuple:
    """Non-power-of-two dims -- exposes tile boundary issues."""
    A_out = torch.randn(333, 257, device=A.device, dtype=A.dtype)
    B_out = torch.randn(257, 129, device=A.device, dtype=A.dtype)
    return A_out, B_out


class MatmulSpec(MatmulKernelSpec):
    name: str = "matmul"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("output_shape",         _wrap(check_output_shape)),
            ("distributivity",       _wrap_dist(check_distributivity)),
            ("scalar_associativity", _wrap(check_scalar_associativity)),
            ("precision_coercion",   _wrap(check_precision_coercion)),
        ]

    @property
    def valid_shapes(self):
        # (M, K, N)
        return [
            (512, 512, 512),
            (256, 512, 1024),
            (1,   512, 512),
            (333, 257, 129),
            (2048, 128, 64),
        ]

    def get_adversarial_inputs(self, inputs):
        A, B = inputs
        return [
            ("k_dim_accumulation",    _k_dim_accumulation(A, B)),
            ("non_aligned_shape",     _non_aligned_shape(A, B)),
            ("swapped_strides",       _swapped_strides_detector(A, B)),
            ("extreme_dynamic_range", _extreme_dynamic_range(A, B)),
            ("identity_weight",       _identity_weight(A, B)),
            ("non_power_of_two",      _non_power_of_two(A, B)),
        ]


def get_spec() -> MatmulSpec:
    return MatmulSpec(name="matmul")


def _wrap(fn):
    def wrapped(candidate_fn, inputs):
        A, B = inputs
        return fn(candidate_fn, A, B)
    return wrapped

def _wrap_dist(fn):
    def wrapped(candidate_fn, inputs):
        A, B = inputs
        C = torch.randn_like(B)
        return fn(candidate_fn, A, B, C)
    return wrapped
