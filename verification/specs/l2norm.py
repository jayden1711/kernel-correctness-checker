"""KernelSpec for l2norm  f(x) -> Tensor, x shape (n_rows, n_cols)."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.norm_properties import (
    check_unit_l2_norm,
    check_positive_scale_invariance,
)


class L2NormSpec(SingleTensorSpec):
    name: str = "l2norm"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("unit_l2_norm", _wrap_output(check_unit_l2_norm)),
            ("positive_scale_invariance", check_positive_scale_invariance),
        ]

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs
        n_cols = x.shape[-1]
        high_variance = x * torch.tensor(
            [0.01, 100.0] * (n_cols // 2) + [0.01] * (n_cols % 2), device=x.device)
        return [("high_variance_row", high_variance)]


def get_spec() -> L2NormSpec:
    return L2NormSpec(name="l2norm")


def _wrap_output(check_fn):
    def wrapped(candidate_fn, inputs):
        return check_fn(candidate_fn(inputs))
    return wrapped
