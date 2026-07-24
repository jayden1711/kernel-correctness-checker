"""
KernelSpec for instancenorm  f(x, weight, bias) -> Tensor.
Reuses LayernormKernelSpec's run_candidate/run_reference/primary_input
unchanged (identical 3-arg signature) -- only make_inputs, valid_shapes,
adversarial inputs, and algebraic_properties differ, since InstanceNorm
operates on 4D (N,C,H,W) tensors reduced per (n,c) over spatial dims,
not LayerNorm's 2D (n_rows,n_cols) reduced over the last dim.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import LayernormKernelSpec
from verification.layer3_properties.instancenorm_properties import (
    check_zero_mean,
    check_unit_variance,
    check_positive_scale_invariance,
)


class InstanceNormSpec(LayernormKernelSpec):
    name: str = "instancenorm"
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
        # (N, C, H, W)
        return [(4, 8, 16, 16), (2, 4, 32, 32), (1, 8, 8, 8), (4, 16, 5, 7)]

    def get_adversarial_inputs(self, inputs):
        x, weight, bias = inputs
        near_const = torch.full_like(x, 3.0) + x * 1e-6
        return [("near_zero_variance", (near_const, weight, bias))]

    def make_inputs(self, shape, device, dtype):
        N, C, H, W = shape
        x = torch.randn(N, C, H, W, device=device, dtype=dtype)
        weight = torch.ones(C, device=device, dtype=dtype)
        bias = torch.zeros(C, device=device, dtype=dtype)
        return x, weight, bias


def get_spec() -> InstanceNormSpec:
    return InstanceNormSpec(name="instancenorm")


def _wrap_identity(check_fn):
    """Run with weight=1, bias=0 so output == normalized x -- required
    for zero_mean/unit_variance to hold (see layernorm.py's own
    identical pattern)."""
    def wrapped(candidate_fn, inputs):
        x, weight, bias = inputs
        ones = torch.ones_like(weight)
        zeros = torch.zeros_like(bias)
        out = candidate_fn(x, ones, zeros)
        return check_fn(out)
    return wrapped
