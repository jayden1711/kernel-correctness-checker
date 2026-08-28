"""KernelSpec for leaky_relu — f(x, negative_slope) -> Tensor

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "leaky_relu";
it is derivation-verified against autograd but NOT probe-verified on a real
Triton kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import KernelSpec


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
class LeakyReluSpec(KernelSpec):
    name: str = "leaky_relu"
    requires_backward: bool = False

    # f(x, negative_slope) -- negative_slope is a fixed python float held constant
    # automatically, since checker.run() only ever replaces inputs[0].
    @property
    def batch_samples(self) -> bool:
        return True

    def run_candidate(self, candidate_fn, inputs):
        x, negative_slope = inputs
        return candidate_fn(x, negative_slope)

    def run_reference(self, reference_fn, inputs):
        x, negative_slope = inputs
        return reference_fn(x, negative_slope)

    def primary_input(self, inputs):
        return inputs[0]

    def make_inputs(self, shape, device, dtype):
        """shape is (*tensor_shape, negative_slope).

        The negative_slope is VARIED, not fixed at the torch default. Fixing it was a
        real hole: a kernel that ignores the argument and hardcodes the default
        is then bit-identical to the reference on every generated input, and no
        check in the pipeline can see it. Measured -- the "hardcoded default"
        mutant escaped the whole battery until this changed.
        """
        *tensor_shape, negative_slope = shape
        return torch.randn(*tensor_shape, device=device, dtype=dtype), negative_slope

    @property
    def valid_shapes(self):
        return [(4096, 0.2), (1024, 0.01), (100000, 0.5), (1, 0.05), (333, 0.3)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # The kink.
            ("exact_zero", _pack(torch.zeros_like(x))),
            # Isolates the negative branch: a kernel that hardcodes the default 0.01 instead of reading the slope argument is only visible here.
            ("all_negative", _pack(-x.abs() - 1.0)),
            # Linear both sides; must scale exactly.
            ("large_magnitude", _pack(x * 1e4)),
        ]


def get_spec() -> LeakyReluSpec:
    return LeakyReluSpec(name="leaky_relu")
