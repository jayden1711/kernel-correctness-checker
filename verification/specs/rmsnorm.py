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


def _non_pow2_width(n_cols: int) -> int:
    """Largest non-power-of-two width w <= min(333, n_cols).

    FIXED 2026-08-28 (verification_runs/oob_adjudication_2026-08-28/): the
    variant used to hardcode width 333 over the CAPTURED gamma, so a harness
    with base width < 333 made the kernel read past gamma's allocation --
    byte-level-proven, and the reason this class's banked records were
    bit-identical across runs (the response was dominated by stable
    out-of-bounds lanes, not by the varying in-bounds gamma)."""
    w = min(333, n_cols)
    while w > 1 and (w & (w - 1)) == 0:   # power of two -> step down
        w -= 1
    return w


def _non_power_of_two(x: torch.Tensor) -> torch.Tensor:
    """Non-power-of-two hidden dimension — exposes tile-boundary bugs.

    DRAW-THEN-SLICE, deliberately: the (n_rows, 333) draw is kept even when
    the target width is smaller so the per-check-reseeded torch stream is
    consumed exactly as before -- _constant_rows and _large_variance draw
    AFTER this variant, and drawing the target width directly would shift
    their inputs and churn records a shape fix has no business touching. At
    base widths >= 333 the slice is a no-op (bitwise pre-fix behaviour)."""
    n_rows, n_cols = x.shape[0], x.shape[-1]
    w = _non_pow2_width(n_cols)
    full = torch.randn(n_rows, 333, device=x.device, dtype=x.dtype)
    return full[:, :w].contiguous() if w < 333 else full


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
        """Return (name, (adv_x, gamma)) pairs. Spec adds gamma.

        gamma is SLICED to each variant's own width (no-op for the
        shape-preserving variants) so every tuple satisfies the contract
        gamma length == adv_x.shape[-1]. Pre-2026-08-28 the width-333
        variant rode on the full-length captured gamma -- the out-of-bounds
        construction proven in
        verification_runs/oob_adjudication_2026-08-28/FINDINGS.md."""
        x, gamma = inputs
        adv_xs = [
            ("large_magnitude",  _large_magnitude(x)),
            ("near_zero",        _near_zero(x)),
            ("non_power_of_two", _non_power_of_two(x)),
            ("constant_rows",    _constant_rows(x)),
            ("large_variance",   _large_variance(x)),
        ]
        return [(name, (adv_x, gamma[: adv_x.shape[-1]]))
                for name, adv_x in adv_xs]


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
    """check_precision_coercion's reference is x / sqrt(mean(x^2)+eps)
    with no gamma multiply (implicitly gamma=1) -- so the candidate must
    be called with identity gamma too, same as _wrap_identity above.
    FIXED: this used to pass through whatever gamma the caller captured,
    which made the check fail on a CORRECT kernel any time gamma wasn't
    already ~1 -- comparing gamma*rmsnorm(x) against bare rmsnorm(x).
    Precision coercion (fp16 in the RMS step) is orthogonal to gamma
    correctness (covered separately by check_gamma_correctness), so
    isolating it via identity gamma is correct, not just a workaround.
    """
    def wrapped(candidate_fn, inputs):
        x, gamma = inputs
        ones = torch.ones_like(gamma)
        fn = lambda xi: candidate_fn(xi, ones.to(xi.dtype))
        return check_fn(fn, x)
    return wrapped
