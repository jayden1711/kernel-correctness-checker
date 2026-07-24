"""KernelSpec for l1norm  f(x) -> Tensor, x shape (n_rows, n_cols)."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.norm_properties import (
    check_unit_l1_norm,
    check_positive_scale_invariance,
)


class L1NormSpec(SingleTensorSpec):
    name: str = "l1norm"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("unit_l1_norm", _wrap_output(check_unit_l1_norm)),
            ("positive_scale_invariance", check_positive_scale_invariance),
        ]

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs
        return [
            ("second_half_dominant", torch.cat([torch.zeros_like(x[:, :x.shape[1]//2]),
                                                  x[:, x.shape[1]//2:] * 10], dim=1)),
        ]


def get_spec() -> L1NormSpec:
    return L1NormSpec(name="l1norm")


def _wrap_output(check_fn):
    def wrapped(candidate_fn, inputs):
        return check_fn(candidate_fn(inputs))
    return wrapped
