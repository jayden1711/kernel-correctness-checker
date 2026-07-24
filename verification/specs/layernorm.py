"""
KernelSpec for layernorm — f(x, gamma, beta) -> Tensor.

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


# Adversarial input generators -- inlined verbatim from the former
# verification/layer2_numeric_oracle/adversarial/layernorm_adversarial.py
# (logic unchanged, only relocated). Return adversarial x tensors only;
# get_adversarial_inputs below wraps each with the captured gamma/beta.

def _skip_mean_subtract(x: torch.Tensor) -> torch.Tensor:
    """
    Large per-row mean shift.
    skip_mean_subtract.py divides raw x by std -- output mean won't be
    zero. wrong_variance_estimate.py also diverges when mean >> 0.
    """
    shifts = torch.linspace(100.0, 1000.0, x.shape[0],
                             device=x.device, dtype=x.dtype).unsqueeze(1)
    return torch.randn_like(x) + shifts


def _zero_variance_rows(x: torch.Tensor) -> torch.Tensor:
    """
    Half the rows are constant (zero variance).
    Exposes division-by-zero handling and wrong eps placement.
    """
    result = torch.zeros_like(x)
    n_zero = x.shape[0] // 2
    result[n_zero:] = torch.randn(
        x.shape[0] - n_zero, x.shape[-1], device=x.device, dtype=x.dtype
    )
    return result


def _large_variance(x: torch.Tensor) -> torch.Tensor:
    """
    Very large values -- exposes fp16 overflow in the squaring step of
    wrong_variance_estimate.py (x^2 overflows fp16 when x ~ 1e4).
    """
    return torch.randn_like(x) * 1e4


def _wrong_variance_trigger(x: torch.Tensor) -> torch.Tensor:
    """
    Large mean with moderate variance -- maximises the numerical
    difference between E[(x-mean)^2] and E[x^2] - mean^2. This is the
    exact condition under which wrong_variance_estimate.py fails.
    """
    mean_val = 1000.0
    return torch.randn_like(x) + mean_val


def _non_power_of_two(x: torch.Tensor) -> torch.Tensor:
    """Non-power-of-two hidden dimension -- exposes tile-boundary bugs."""
    n_rows = x.shape[0]
    return torch.randn(n_rows, 333, device=x.device, dtype=x.dtype)


class LayernormSpec(LayernormKernelSpec):
    name: str = "layernorm"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("zero_mean",          _wrap_identity(check_zero_mean)),
            ("unit_variance",      _wrap_identity(check_unit_variance)),
            ("scale_invariance",   _wrap_scale(check_scale_invariance)),
            ("affine_correctness", _wrap_affine(check_affine_correctness)),
            ("precision_coercion", _wrap_precision(check_precision_coercion)),
        ]

    @property
    def valid_shapes(self):
        return [
            (512,  512),
            (256,  1024),
            (1,    512),
            (1000, 333),
            (2048, 128),
        ]

    def get_adversarial_inputs(self, inputs):
        """Return (name, (adv_x, gamma, beta)) pairs -- gamma/beta held
        fixed at whatever was captured, only x varies."""
        x, gamma, beta = inputs
        adv_xs = [
            ("skip_mean_subtract",     _skip_mean_subtract(x)),
            ("zero_variance_rows",     _zero_variance_rows(x)),
            ("large_variance",         _large_variance(x)),
            ("wrong_variance_trigger", _wrong_variance_trigger(x)),
            ("non_power_of_two",       _non_power_of_two(x)),
        ]
        return [(name, (adv_x, gamma, beta)) for name, adv_x in adv_xs]


def get_spec() -> LayernormSpec:
    return LayernormSpec(name="layernorm")


def _wrap_identity(check_fn):
    def wrapped(candidate_fn, inputs):
        x, gamma, beta = inputs
        ones  = torch.ones_like(gamma)
        zeros = torch.zeros_like(beta)
        out = candidate_fn(x, ones, zeros)
        return check_fn(out)
    return wrapped


def _wrap_scale(check_fn):
    def wrapped(candidate_fn, inputs):
        x, gamma, beta = inputs
        fn = lambda xi: candidate_fn(xi, gamma, beta)
        return check_fn(fn, x)
    return wrapped


def _wrap_precision(check_fn):
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
