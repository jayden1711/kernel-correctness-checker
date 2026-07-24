"""
KernelSpec for flash attention — f(Q, K, V) -> Tensor.
All inputs are 2D: shape (N, D). No batch or head dimensions.
This matches the actual TritonBench reference and cheating kernels.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Callable
import torch

from verification.specs.base_spec import AttentionKernelSpec
from verification.layer3_properties.flash_attention_properties import (
    check_output_bounded_by_values,
    check_attention_weights_sum_to_one,
    check_precision_coercion,
)


# Adversarial input generators -- inlined verbatim from the former
# verification/layer2_numeric_oracle/adversarial/flash_attention_adversarial.py
# (logic unchanged, only relocated). Targets each cheating kernel directly:
#   approx_denom.py     incomplete normalizer (first half of tiles only)
#   drop_last_tile.py   loop stops at N - BLOCK_N, skips last tile
#   skip_rescaling.py   omits exp(m - m_new) rescaling of acc and l
#   wrong_mask.py       causal mask off-by-one (q_idx > kv_idx + 1)

def _make_qkv(N, D, device, dtype):
    Q = torch.randn(N, D, device=device, dtype=dtype)
    K = torch.randn(N, D, device=device, dtype=dtype)
    V = torch.randn(N, D, device=device, dtype=dtype)
    return Q, K, V


def _approx_denominator(N=128, D=64, device="cpu", dtype=torch.float32) -> tuple:
    """
    Large Q values so that skipping subtract-max causes large softmax error.
    Targets approx_denom.py: normalizer only updated for first half of tiles.
    """
    Q, K, V = _make_qkv(N, D, device, dtype)
    Q = Q * 10.0
    return Q, K, V


def _last_tile_dropped(N=65, D=64, device="cpu", dtype=torch.float32) -> tuple:
    """
    N not a multiple of BLOCK_N=32 (65 = 2x32 + 1).
    High-value tokens in last position maximise the error from drop_last_tile.py.
    """
    Q, K, V = _make_qkv(N, D, device, dtype)
    K[-1, :] = 1e4
    V[-1, :] = 1e4
    return Q, K, V


def _multi_tile_rescaling(N=192, D=64, device="cpu", dtype=torch.float32) -> tuple:
    """
    N = 6 x BLOCK_N=32, forces 6 tile iterations.
    Max score shifts dramatically between tiles -- rescaling error compounds.
    Targets skip_rescaling.py.
    """
    BLOCK = 32
    Q, K, V = _make_qkv(N, D, device, dtype)
    K[:BLOCK, :]        *= 1e-6   # tile 1: near-zero
    K[BLOCK:BLOCK*2, :]  *= 1.0   # tile 2: normal
    K[BLOCK*2:, :]       *= 1e4   # tiles 3-6: very large -- running max jumps
    return Q, K, V


def _skip_rescaling(N=128, D=64, device="cpu", dtype=torch.float32) -> tuple:
    """
    Max score shifts between first and second half of sequence.
    Targets skip_rescaling.py directly.
    """
    Q, K, V = _make_qkv(N, D, device, dtype)
    K[:N // 2, :] *= 1e-6
    K[N // 2:, :] *= 1e4
    return Q, K, V


def _wrong_causal_mask(N=128, D=64, device="cpu", dtype=torch.float32) -> tuple:
    """
    Future positions have very large K values.
    wrong_mask.py uses q_idx > kv_idx + 1, masking self-attention.
    This makes the off-by-one error large and measurable.
    """
    Q, K, V = _make_qkv(N, D, device, dtype)
    # wrong mask blocks these, correct mask allows them
    for i in range(N):
        K[i, :] = 1e4
        break  # just first row is enough to expose the bug
    return Q, K, V


def _equal_attention_weights(N=128, D=64, device="cpu", dtype=torch.float32) -> tuple:
    """
    Q=K=ones -> uniform attention weights.
    Output[i, :] must equal mean(V, dim=0) for all i.
    Any normalizer bug produces wrong scale.
    """
    Q = torch.ones(N, D, device=device, dtype=dtype)
    K = torch.ones(N, D, device=device, dtype=dtype)
    V = torch.randn(N, D, device=device, dtype=dtype)
    return Q, K, V


class FlashAttentionSpec(AttentionKernelSpec):
    name: str = "flash_attention"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("output_bounded_by_values",     _wrap_bounded(check_output_bounded_by_values)),
            ("attention_weights_sum_to_one",  _wrap_attn_sum(check_attention_weights_sum_to_one)),
            ("precision_coercion",            _wrap_precision(check_precision_coercion)),
        ]

    @property
    def valid_shapes(self):
        return [
            (128,  64),   # base case
            (64,   64),   # small
            (256,  64),   # larger sequence
            (65,   64),   # non-power-of-two N (exposes drop_last_tile)
            (192,  64),   # 3x BLOCK_N=32, forces 6 tile iterations
        ]

    def get_adversarial_inputs(self, inputs):
        """Return all adversarial variants as (name, (Q, K, V)) pairs."""
        Q, K, V = inputs
        device, dtype = Q.device, Q.dtype
        N, D = Q.shape
        return [
            ("approx_denominator",      _approx_denominator(N, D, device, dtype)),
            ("last_tile_dropped",       _last_tile_dropped(65, D, device, dtype)),
            ("multi_tile_rescaling",    _multi_tile_rescaling(192, D, device, dtype)),
            ("skip_rescaling",          _skip_rescaling(N, D, device, dtype)),
            ("wrong_causal_mask",       _wrong_causal_mask(N, D, device, dtype)),
            ("equal_attention_weights", _equal_attention_weights(N, D, device, dtype)),
        ]


def get_spec() -> FlashAttentionSpec:
    return FlashAttentionSpec(name="flash_attention")


def _wrap_bounded(fn):
    def wrapped(candidate_fn, inputs):
        Q, K, V = inputs
        out = candidate_fn(Q, K, V)
        return fn(out, V)
    return wrapped

def _wrap_attn_sum(fn):
    def wrapped(candidate_fn, inputs):
        Q, K, V = inputs
        return fn(candidate_fn, Q, K, V)
    return wrapped

def _wrap_precision(fn):
    def wrapped(candidate_fn, inputs):
        Q, K, V = inputs
        return fn(candidate_fn, Q, K, V)
    return wrapped
