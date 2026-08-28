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


def _non_pow2_width(n_cols: int) -> int:
    """Largest non-power-of-two width w <= min(333, n_cols).

    FIXED 2026-08-28 (verification_runs/oob_adjudication_2026-08-28/): the
    variant used to hardcode width 333 while get_adversarial_inputs re-wraps
    the CAPTURED companions, so any harness whose base width is < 333 fed
    the kernel a gamma/beta shorter than n_cols -- and the kernel then read
    205 floats past the allocation (byte-level-proven: the outputs contained
    the contents of adjacent tensors). The width now adapts to the base
    shape so the sliced companions below always satisfy the op's contract."""
    w = min(333, n_cols)
    while w > 1 and (w & (w - 1)) == 0:   # power of two -> step down
        w -= 1
    return w


def _non_power_of_two(x: torch.Tensor) -> torch.Tensor:
    """Non-power-of-two hidden dimension -- exposes tile-boundary bugs.

    DRAW-THEN-SLICE, deliberately: the (n_rows, 333) draw is kept even when
    the target width is smaller, so this variant consumes exactly as much of
    the per-check-reseeded torch stream as the pre-fix code did. Drawing the
    target width directly would shift every draw made after this variant
    (rmsnorm generates constant_rows and large_variance AFTER non_pow2) and
    turn a shape fix into unrelated record churn. At base widths >= 333 the
    slice is a no-op and the variant is bitwise identical to its pre-fix
    behaviour."""
    n_rows, n_cols = x.shape[0], x.shape[-1]
    w = _non_pow2_width(n_cols)
    full = torch.randn(n_rows, 333, device=x.device, dtype=x.dtype)
    return full[:, :w].contiguous() if w < 333 else full


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
        fixed at whatever was captured, only x varies.

        Companions are SLICED to each variant's own width (a no-op for the
        shape-preserving variants) so every returned tuple satisfies the
        contract stated at the top of this file: gamma/beta length ==
        adv_x.shape[-1]. Before 2026-08-28 the width-333 variant rode on
        full-length captured companions, which is exactly the out-of-bounds
        construction the adjudication round proved -- see
        verification_runs/oob_adjudication_2026-08-28/FINDINGS.md."""
        x, gamma, beta = inputs
        adv_xs = [
            ("skip_mean_subtract",     _skip_mean_subtract(x)),
            ("zero_variance_rows",     _zero_variance_rows(x)),
            ("large_variance",         _large_variance(x)),
            ("wrong_variance_trigger", _wrong_variance_trigger(x)),
            ("non_power_of_two",       _non_power_of_two(x)),
        ]
        return [(name, (adv_x,
                        gamma[: adv_x.shape[-1]],
                        beta[: adv_x.shape[-1]]))
                for name, adv_x in adv_xs]


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
    """check_precision_coercion's reference is F.layer_norm(x) with no
    affine transform (implicitly gamma=1, beta=0) -- so the candidate must
    be called with identity gamma/beta too, same as _wrap_identity above.
    FIXED: this used to pass through whatever gamma/beta the caller
    captured, which made the check fail on a CORRECT kernel any time
    gamma/beta weren't already ~(1, 0) -- comparing gamma*norm(x)+beta
    against bare norm(x). Precision coercion (fp16 in the variance step)
    is orthogonal to affine correctness (covered separately by
    check_affine_correctness), so isolating it via identity gamma/beta
    is correct, not just a workaround.
    """
    def wrapped(candidate_fn, inputs):
        x, gamma, beta = inputs
        ones = torch.ones_like(gamma)
        zeros = torch.zeros_like(beta)
        fn = lambda xi: candidate_fn(xi, ones.to(xi.dtype), zeros.to(xi.dtype))
        return check_fn(fn, x)
    return wrapped


def _wrap_affine(check_fn):
    def wrapped(candidate_fn, inputs):
        x, gamma, beta = inputs
        return check_fn(candidate_fn, x)
    return wrapped
