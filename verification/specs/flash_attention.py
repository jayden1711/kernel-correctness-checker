"""
KernelSpec for flash attention  f(Q, K, V) -> Tensor.
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
from verification.layer2_numeric_oracle.adversarial.flash_attention_adversarial import (
    get_adversarial_inputs as _get_adversarial,
)


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
        Q, K, V = inputs
        return _get_adversarial(Q, K, V)


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