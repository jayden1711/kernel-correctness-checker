"""KernelSpec for mse_loss — f(x, target) -> scalar mean squared error

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "mse_loss";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import TargetLossKernelSpec


@dataclass
class MseLossSpec(TargetLossKernelSpec):
    name: str = "mse_loss"
    requires_backward: bool = False

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # Loss must be exactly 0. Catches an uninitialised accumulator, which a nonzero random loss would hide completely.
            ("identical_to_target", _pack(rest[0].clone() if rest else x)),
            # Squaring overflows fp16 long before the input does.
            ("large_magnitude", _pack(x * 1e4)),
            # Squared differences underflow to zero in fp32; the correct loss is small but nonzero.
            ("tiny_differences", _pack((rest[0] + x * 1e-7) if rest else x * 1e-7)),
        ]


def get_spec() -> MseLossSpec:
    return MseLossSpec(name="mse_loss")
