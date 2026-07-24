"""KernelSpec for log_softmax  f(x) -> Tensor, x shape (n_rows, n_cols)."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.log_softmax_properties import (
    check_exp_sums_to_one,
    check_shift_invariance,
    check_monotonicity,
)


class LogSoftmaxSpec(SingleTensorSpec):
    name: str = "log_softmax"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("exp_sums_to_one", _wrap_output(check_exp_sums_to_one)),
            ("shift_invariance", check_shift_invariance),
            ("monotonicity", check_monotonicity),
        ]

    @property
    def valid_shapes(self):
        return [
            (512, 512),
            (256, 1024),
            (1, 512),
            (1000, 333),
            (2048, 128),
        ]

    def get_adversarial_inputs(self, inputs):
        # New, minimal generator -- not adapted from an existing file
        # (never seen softmax_adversarial.py's actual contents).
        x = inputs
        return [
            ("large_magnitude", x * 200 + 100),
            ("near_zero_variance", torch.full_like(x, 3.0) + x * 1e-6),
        ]


def get_spec() -> LogSoftmaxSpec:
    return LogSoftmaxSpec(name="log_softmax")


def _wrap_output(check_fn):
    def wrapped(candidate_fn, inputs):
        out = candidate_fn(inputs)
        return check_fn(out)
    return wrapped
