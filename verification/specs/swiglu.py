"""KernelSpec for swiglu — f(x) -> silu(x[..., :h]) * x[..., h:]

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "swiglu";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec


@dataclass
class SwigluSpec(SingleTensorSpec):
    name: str = "swiglu"
    requires_backward: bool = False

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (333, 128), (2048, 64)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # silu gate underflows to 0, so the output must be exactly 0 regardless of the linear half. Catches a kernel that swapped the two halves -- the single most common SwiGLU bug, and invisible on symmetric random data.
            ("gate_saturating_neg", _pack(torch.cat([torch.full_like(x[..., :x.shape[-1]//2], -40.0), x[..., x.shape[-1]//2:]], dim=-1))),
            # silu(a) -> a, so output -> a * b with a = 40 exactly.
            ("gate_saturating_pos", _pack(torch.cat([torch.full_like(x[..., :x.shape[-1]//2], 40.0), x[..., x.shape[-1]//2:]], dim=-1))),
            # Output must be identically zero -- the other half-swap direction.
            ("linear_half_zero", _pack(torch.cat([x[..., :x.shape[-1]//2], torch.zeros_like(x[..., x.shape[-1]//2:])], dim=-1))),
        ]


def get_spec() -> SwigluSpec:
    return SwigluSpec(name="swiglu")
