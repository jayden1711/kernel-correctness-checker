"""KernelSpec for huber_loss — f(x, target) -> scalar smooth-L1 / Huber loss

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "huber_loss";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import TargetLossKernelSpec


@dataclass
class HuberLossSpec(TargetLossKernelSpec):
    name: str = "huber_loss"
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
            # Places every residual exactly at |d| = beta = 1, the kink where the quadratic and linear branches join. An off-by-one comparison (< vs <=) is only visible here.
            ("at_beta_boundary", _pack((rest[0] + torch.sign(x)) if rest else x)),
            # All residuals deep in the linear branch -- catches a kernel that only implemented the quadratic half, which random data mostly exercises.
            ("far_linear_regime", _pack((rest[0] + torch.sign(x) * 1e3) if rest else x * 1e3)),
            # Loss must be exactly 0.
            ("identical_to_target", _pack(rest[0].clone() if rest else x)),
        ]


def get_spec() -> HuberLossSpec:
    return HuberLossSpec(name="huber_loss")
