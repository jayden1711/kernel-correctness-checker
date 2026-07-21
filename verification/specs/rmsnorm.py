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
from verification.layer2_numeric_oracle.adversarial.rmsnorm_adversarial import (
    get_adversarial_inputs as _get_adversarial,
)


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
        x, gamma = inputs
        adv_xs = _get_adversarial(x)
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