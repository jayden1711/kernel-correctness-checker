"""KernelSpec for softmax  f(x) -> Tensor, x shape (n_rows, n_cols)."""

from dataclasses import dataclass, field
from typing import List, Tuple, Callable
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.softmax_properties import (
    check_rows_sum_to_one,
    check_shift_invariance,
    check_monotonicity,
    check_precision_coercion,
)
from verification.layer2_numeric_oracle.adversarial.softmax_adversarial import (
    get_adversarial_inputs as _get_adversarial,
)


class SoftmaxSpec(SingleTensorSpec):
    name: str = "softmax"
    requires_backward: bool = False  # reference softmax kernel has no autograd

    @property
    def algebraic_properties(self):
        return [
            ("rows_sum_to_one",    _wrap_output(check_rows_sum_to_one)),
            ("shift_invariance",   check_shift_invariance),
            ("monotonicity",       check_monotonicity),
            ("precision_coercion", check_precision_coercion),
        ]

    @property
    def valid_shapes(self):
        return [
            (512, 512),
            (256, 1024),
            (1,   512),
            (1000, 333),
            (2048, 128),
        ]

    def get_adversarial_inputs(self, inputs):
        return _get_adversarial(inputs)


def get_spec() -> SoftmaxSpec:
    return SoftmaxSpec(name="softmax")


def _wrap_output(check_fn):
    def wrapped(candidate_fn, inputs):
        out = candidate_fn(inputs)
        return check_fn(out)
    return wrapped