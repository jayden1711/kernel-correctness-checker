"""KernelSpec for rmsnorm — f(x, gamma) -> Tensor, x:(n_rows, n_cols), gamma:(n_cols,)."""

from dataclasses import dataclass
from typing import List, Tuple, Callable
import torch

from verification.specs.base_spec import RMSNormKernelSpec
from verification.layer3_properties.rmsnorm_properties import (
    check_unit_rms,
    check_scale_invariance,
    check_gamma_correctness,
    check_precision_coercion,
)


# Adversarial input generators -- inlined verbatim from the former
# verification/layer2_numeric_oracle/adversarial/rmsnorm_adversarial.py
# (logic unchanged, only relocated). Return adversarial x tensors only;
# get_adversarial_inputs below wraps each with the captured gamma.

def _large_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    Very large values — exposes fp16 overflow in the x^2 step.
    sqrt(mean(x^2)) overflows fp16 when x ~ 1e4.
    """
    return torch.randn_like(x) * 1e4


def _near_zero(x: torch.Tensor) -> torch.Tensor:
    """
    Near-zero input — tests eps handling.
    A kernel that omits eps will divide by ~0 and produce Inf/NaN.
    """
    return torch.randn_like(x) * 1e-8


def _non_power_of_two(x: torch.Tensor) -> torch.Tensor:
    """Non-power-of-two hidden dimension — exposes tile-boundary bugs."""
    n_rows = x.shape[0]
    return torch.randn(n_rows, 333, device=x.device, dtype=x.dtype)


def _constant_rows(x: torch.Tensor) -> torch.Tensor:
    """
    All elements in each row identical (but different across rows).
    Output should be gamma * sign(x) for nonzero x (since x/RMS(x) = sign(x)
    when all elements are equal).
    Catches kernels with wrong reduction axis.
    """
    vals = torch.randn(x.shape[0], 1, device=x.device, dtype=x.dtype) * 10.0
    return vals.expand_as(x).contiguous()


def _large_variance(x: torch.Tensor) -> torch.Tensor:
    """
    Extreme spread — first half of columns near zero, second half very large.
    Catches partial_reduction: if only the first half is reduced, the RMS
    is near zero, inflating the output wildly.
    """
    result = torch.zeros_like(x)
    mid = x.shape[-1] // 2
    result[:, mid:] = torch.randn(x.shape[0], x.shape[-1] - mid,
                                   device=x.device, dtype=x.dtype) * 1e4
    return result


class RMSNormSpec(RMSNormKernelSpec):
    name: str = "rmsnorm"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("unit_rms",           _wrap_identity(check_unit_rms)),
            ("scale_invariance",   _wrap_scale(check_scale_invariance)),
            ("gamma_correctness",  _wrap_gamma(check_gamma_correctness)),
            ("precision_coercion", _wrap_precision(check_precision_coercion)),
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
        """Return (name, (adv_x, gamma)) pairs. Spec adds gamma."""
        x, gamma = inputs
        adv_xs = [
            ("large_magnitude",  _large_magnitude(x)),
            ("near_zero",        _near_zero(x)),
            ("non_power_of_two", _non_power_of_two(x)),
            ("constant_rows",    _constant_rows(x)),
            ("large_variance",   _large_variance(x)),
        ]
        return [(name, (adv_x, gamma)) for name, adv_x in adv_xs]


def get_spec() -> RMSNormSpec:
    return RMSNormSpec(name="rmsnorm")


# Wrappers — same pattern as layernorm spec

def _wrap_identity(check_fn):
    """Run with gamma=1 so output == x / RMS(x)."""
    def wrapped(candidate_fn, inputs):
        x, gamma = inputs
        ones = torch.ones_like(gamma)
        out = candidate_fn(x, ones)
        return check_fn(out)
    return wrapped


def _wrap_scale(check_fn):
    def wrapped(candidate_fn, inputs):
        x, gamma = inputs
        fn = lambda xi: candidate_fn(xi, gamma)
        return check_fn(fn, x)
    return wrapped


def _wrap_gamma(check_fn):
    def wrapped(candidate_fn, inputs):
        x, gamma = inputs
        return check_fn(candidate_fn, x)
    return wrapped


def _wrap_precision(check_fn):
    def wrapped(candidate_fn, inputs):
        x, gamma = inputs
        fn = lambda xi: candidate_fn(xi, gamma.to(xi.dtype))
        return check_fn(fn, x)
    return wrapped
