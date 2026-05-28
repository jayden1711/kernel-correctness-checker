"""KernelSpec for matmul  f(A, B) -> Tensor, A:(M,K), B:(K,N)."""

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
from verification.layer2_numeric_oracle.adversarial.matmul_adversarial import (
    get_adversarial_inputs as _get_adversarial,
)


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
        return _get_adversarial(A, B)


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