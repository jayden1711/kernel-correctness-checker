"""
KernelSpec for layernorm  f(x, gamma, beta) -> Tensor.

inputs tuple: (x, gamma, beta)
  x:     (n_rows, n_cols)
  gamma: (n_cols,)
  beta:  (n_cols,)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Callable
import torch

from verification.specs.base_spec import LayernormKernelSpec
from verification.layer3_properties.layernorm_properties import (
    check_zero_mean,
    check_unit_variance,
    check_scale_invariance,
    check_precision_coercion,
    check_affine_correctness
)
from verification.layer2_numeric_oracle.adversarial.layernorm_adversarial import (
    get_adversarial_inputs as _get_adversarial,
)


class LayernormSpec(LayernormKernelSpec):
    name: str = "layernorm"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            # zero_mean and unit_variance only hold when gamma=1, beta=0
            # We run them on the normalised output with identity affine params
            ("zero_mean",          _wrap_identity(check_zero_mean)),
            ("unit_variance",      _wrap_identity(check_unit_variance)),
            ("scale_invariance",   _wrap_scale(check_scale_invariance)),
            ("affine_correctness", _wrap_affine(check_affine_correctness)),
            ("precision_coercion", _wrap_precision(check_precision_coercion)),
        ]

    @property
    def valid_shapes(self):
        # shapes are (n_rows, n_cols); make_inputs builds gamma/beta automatically
        return [
            (512,  512),
            (256,  1024),
            (1,    512),
            (1000, 333),
            (2048, 128),
        ]

    def get_adversarial_inputs(self, inputs):
        x, gamma, beta = inputs
        # adversarial generators vary x; keep same gamma/beta
        adv_xs = _get_adversarial(x)
        return [(name, (adv_x, gamma, beta)) for name, adv_x in adv_xs]


def get_spec() -> LayernormSpec:
    return LayernormSpec(name="layernorm")



def _wrap_identity(check_fn):
    """
    Run with gamma=1, beta=0 so output == normalised x.
    zero_mean and unit_variance only hold for identity affine.
    """
    def wrapped(candidate_fn, inputs):
        x, gamma, beta = inputs
        ones  = torch.ones_like(gamma)
        zeros = torch.zeros_like(beta)
        out = candidate_fn(x, ones, zeros)
        return check_fn(out)
    return wrapped


def _wrap_scale(check_fn):
    """check_scale_invariance(kernel_fn, x)  pass x only."""
    def wrapped(candidate_fn, inputs):
        x, gamma, beta = inputs
        # wrap candidate so it uses the stored gamma/beta
        fn = lambda xi: candidate_fn(xi, gamma, beta)
        return check_fn(fn, x)
    return wrapped


def _wrap_precision(check_fn):
    """check_precision_coercion(kernel_fn, x)."""
    def wrapped(candidate_fn, inputs):
        x, gamma, beta = inputs
        fn = lambda xi: candidate_fn(xi, gamma.to(xi.dtype), beta.to(xi.dtype))
        return check_fn(fn, x)
    return wrapped

def _wrap_affine(check_fn):
    def wrapped(candidate_fn, inputs):
        x, gamma, beta = inputs
        return check_fn(candidate_fn, x)
    return wrapped