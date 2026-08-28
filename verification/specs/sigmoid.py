"""KernelSpec for sigmoid — f(x) -> Tensor

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "sigmoid";
it is derivation-verified against autograd but NOT probe-verified on a real
Triton kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec


def _out(cf, inputs):
    return cf(*inputs) if isinstance(inputs, tuple) else cf(inputs)


def _check_nonneg(cf, inputs):
    o = _out(cf, inputs)
    bad = (o < 0).sum().item()
    return (bad == 0), (f"{bad} negative output element(s)" if bad
                        else "all outputs non-negative")


def _check_positive(cf, inputs):
    o = _out(cf, inputs)
    bad = (o <= 0).sum().item()
    return (bad == 0), (f"{bad} non-positive output element(s)" if bad
                        else "all outputs strictly positive")


def _check_range(cf, inputs, lo, hi):
    o = _out(cf, inputs)
    bad = ((o < lo - 1e-5) | (o > hi + 1e-5)).sum().item()
    return (bad == 0), (f"{bad} element(s) outside [{lo}, {hi}]" if bad
                        else f"all outputs within [{lo}, {hi}]")


def _check_idempotent(cf, inputs):
    """f(f(x)) == f(x). True for relu; a leaky mutant breaks it."""
    o = _out(cf, inputs)
    o2 = cf(o) if not isinstance(inputs, tuple) else cf(o, *inputs[1:])
    ok = torch.allclose(o, o2, atol=1e-6, rtol=1e-6)
    return ok, ("idempotent" if ok else
                f"f(f(x)) != f(x), max diff {(o - o2).abs().max().item():.3e}")


def _check_odd(cf, inputs):
    """f(-x) == -f(x)."""
    x = inputs[0] if isinstance(inputs, tuple) else inputs
    rest = inputs[1:] if isinstance(inputs, tuple) else ()
    a = cf(x, *rest) if rest else cf(x)
    b = cf(-x, *rest) if rest else cf(-x)
    ok = torch.allclose(a, -b, atol=1e-5, rtol=1e-5)
    return ok, ("odd symmetric" if ok else
                f"f(-x) != -f(x), max diff {(a + b).abs().max().item():.3e}")


@dataclass
class SigmoidSpec(SingleTensorSpec):
    name: str = "sigmoid"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ('range_0_1', lambda cf, inputs: _check_range(cf, inputs, 0.0, 1.0)),
        ]

    @property
    def valid_shapes(self):
        return [(4096,), (1024,), (100000,), (1,), (333,)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # exp(-x) underflows; output must be exactly 1.0, not NaN. Catches the naive 1/(1+exp(-x)) form without the stable branch.
            # 100.0, NOT 40.0: fp32 exp() overflows near x=88, so at 40 (exp(40)=2.4e17,
            # well inside range) a naive unstable formulation stays FINITE and the
            # variant silently tested nothing. Measured -- the unstable-exp mutant
            # escaped the entire battery at 40 and is caught at 100.
            ("saturating_pos", _pack(torch.full_like(x, 100.0) + x * 0.01)),
            # exp(-x) OVERFLOWS in the naive form -- the classic sigmoid kernel bug.
            ("saturating_neg", _pack(torch.full_like(x, -100.0) + x * 0.01)),
            # Output must be ~0.5; catches a dropped +1 in the denominator.
            ("near_zero", _pack(x * 1e-6)),
        ]


def get_spec() -> SigmoidSpec:
    return SigmoidSpec(name="sigmoid")
