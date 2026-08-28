"""KernelSpec for std_reduction — f(x) -> std over last dim (unbiased)

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "std_reduction";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec


def _check_std_nonneg(candidate_fn, inputs):
    y = candidate_fn(inputs)
    bad = (y < -1e-6).sum().item()
    nan = (~torch.isfinite(y)).sum().item()
    if nan:
        return False, f"{nan} non-finite output(s) -- likely sqrt of a negative variance"
    return (bad == 0), (f"{bad} negative std value(s)" if bad else "all std values non-negative")


def _check_shift_invariant(candidate_fn, inputs):
    """std(x + c) == std(x). The property a one-pass kernel silently violates."""
    a = candidate_fn(inputs).float()
    b = candidate_fn(inputs + 1e4).float()
    ok = torch.allclose(a, b, atol=1e-2, rtol=1e-2)
    return ok, ("shift invariant" if ok else
                f"std changed under a constant shift, max diff {(a - b).abs().max().item():.3e}")


@dataclass
class StdReductionSpec(SingleTensorSpec):
    name: str = "std_reduction"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ('nonneg', lambda cf, inputs: _check_std_nonneg(cf, inputs)),
            ('shift_invariant', lambda cf, inputs: _check_shift_invariant(cf, inputs)),
        ]

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # std must be exactly 0. Catches catastrophic cancellation in a naive E[x^2] - E[x]^2 kernel, which returns a small negative number here and then NaNs in the sqrt -- the classic one-pass variance bug.
            ("constant_rows", _pack(torch.ones_like(x))),
            # Shift-invariance: std is unchanged by adding a constant, but a one-pass kernel loses all precision because the mean dominates the sum of squares.
            ("large_offset", _pack(x + 1e6)),
            # One extreme value per row; the answer is dominated by it and easy to verify.
            ("single_outlier", _pack(torch.zeros_like(x).index_fill_(-1, torch.tensor([0], device=x.device), 1e3))),
        ]


def get_spec() -> StdReductionSpec:
    return StdReductionSpec(name="std_reduction")
