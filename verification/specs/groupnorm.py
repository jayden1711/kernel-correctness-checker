"""KernelSpec for groupnorm  f(x, num_groups, weight, bias) -> Tensor."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import GroupNormKernelSpec
from verification.layer3_properties.groupnorm_properties import (
    check_zero_mean,
    check_unit_variance,
    check_positive_scale_invariance,
)


class GroupNormSpec(GroupNormKernelSpec):
    name: str = "groupnorm"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("zero_mean", _wrap_identity(check_zero_mean)),
            ("unit_variance", _wrap_identity(check_unit_variance)),
            ("positive_scale_invariance",
             lambda cf, inputs: check_positive_scale_invariance(cf, *inputs)),
        ]

    @property
    def valid_shapes(self):
        # (N, C, H, W, num_groups)
        return [(4, 8, 16, 16, 4), (2, 4, 32, 32, 2), (1, 8, 8, 8, 4), (4, 16, 5, 7, 8)]

    def get_adversarial_inputs(self, inputs):
        x, num_groups, weight, bias = inputs
        near_const = torch.full_like(x, 3.0) + x * 1e-6
        return [("near_zero_variance", (near_const, num_groups, weight, bias))]


def get_spec() -> GroupNormSpec:
    return GroupNormSpec(name="groupnorm")


def _wrap_identity(check_fn):
    def wrapped(candidate_fn, inputs):
        x, num_groups, weight, bias = inputs
        ones = torch.ones_like(weight)
        zeros = torch.zeros_like(bias)
        out = candidate_fn(x, num_groups, ones, zeros)
        return check_fn(out, num_groups)
    return wrapped
