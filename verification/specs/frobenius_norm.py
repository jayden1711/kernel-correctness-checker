"""KernelSpec for frobenius_norm  f(x) -> Tensor, whole-tensor normalization."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.norm_properties import (
    check_unit_frobenius_norm,
    check_positive_scale_invariance,
)


class FrobeniusNormSpec(SingleTensorSpec):
    # EXPLICIT OVERRIDE back to the safe default. This kernel reduces across
    # the WHOLE tensor rather than within a row (see
    # TritonBench/reference/frobenius_norm.py: "every other operator reduces
    # within one row/instance per program; this one reduces across the WHOLE
    # tensor"). Stacking 20 perturbation samples would compute ONE norm over
    # 20x the data and return it as if it were 20 sensitivities -- a
    # plausible wrong number, not an error. Measured on a stand-in with the
    # real semantics: adaptive_tol went 0.001218 -> 0.778163, a 639x LOOSER
    # tolerance, with no exception raised. Do not remove this.
    @property
    def batch_samples(self) -> bool:
        return False

    name: str = "frobenius_norm"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("unit_frobenius_norm", _wrap_output(check_unit_frobenius_norm)),
            ("positive_scale_invariance", check_positive_scale_invariance),
        ]

    @property
    def valid_shapes(self):
        # Keep these SMALL -- the reference kernel uses O(n) atomic_add
        # over the whole tensor, unlike every other operator here.
        return [(37, 53), (20, 20), (64, 64), (1, 100), (100, 1)]

    def get_adversarial_inputs(self, inputs):
        x = inputs.clone()
        x[0, 0] = 500.0  # one dominant outlier -- high variance
        return [("dominant_outlier", x)]


def get_spec() -> FrobeniusNormSpec:
    return FrobeniusNormSpec(name="frobenius_norm")


def _wrap_output(check_fn):
    def wrapped(candidate_fn, inputs):
        return check_fn(candidate_fn(inputs))
    return wrapped
