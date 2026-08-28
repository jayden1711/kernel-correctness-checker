"""KernelSpec for relu — f(x) -> Tensor, elementwise

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "relu";
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
class ReluSpec(SingleTensorSpec):
    name: str = "relu"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ('nonneg_output', lambda cf, inputs: _check_nonneg(cf, inputs)),
            ('idempotent', lambda cf, inputs: _check_idempotent(cf, inputs)),
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
            # The kink. relu(0)=0 and the subgradient is a convention; a kernel using >= vs > on the wrong side still passes on random data.
            ("exact_zero", _pack(torch.zeros_like(x))),
            # Whole tensor in the dead zone. Output must be identically 0 -- catches a kernel that leaks the input through.
            ("all_negative", _pack(-x.abs() - 1.0)),
            # No saturation in relu, so output must scale exactly linearly.
            ("large_magnitude", _pack(x * 1e4)),
            # Small POSITIVE band. Every other variant is at zero, negative, or
            # large -- so a kernel that zeroes everything below a small epsilon
            # (a real thresholding bug) matched the reference on all of them.
            # Measured: the zero_at_boundary mutant escaped the whole battery
            # until this variant existed.
            ("small_positive", _pack(x.abs() * 1e-3 + 1e-6)),
        ]


def get_spec() -> ReluSpec:
    return ReluSpec(name="relu")
