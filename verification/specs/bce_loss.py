"""KernelSpec for bce_loss — f(p, target) -> scalar binary cross-entropy

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "bce_loss";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import TargetLossKernelSpec



@dataclass
class BceLossSpec(TargetLossKernelSpec):
    name: str = "bce_loss"
    requires_backward: bool = False

    def make_inputs(self, shape, device, dtype):
        """BCE needs probabilities in (0,1) and 0/1 targets, NOT randn.

        Inherited TargetLossKernelSpec.make_inputs returns randn for both,
        which puts p outside [0,1] and makes log(p) NaN for roughly half the
        tensor -- the reference itself would fail before the candidate was
        ever judged. Overridden rather than worked around downstream.
        """
        p = torch.rand(*shape, device=device, dtype=dtype) * 0.98 + 0.01
        t = torch.randint(0, 2, shape, device=device).to(dtype)
        return p, t

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # log(p) -> -inf. The signature BCE bug is a missing clamp; this is where it shows.
            ("near_zero_prob", _pack(torch.full_like(x, 1e-7))),
            # log(1-p) -> -inf, the other side.
            ("near_one_prob", _pack(torch.full_like(x, 1 - 1e-7))),
            # Loss must be exactly log(2).
            ("exact_half", _pack(torch.full_like(x, 0.5))),
            # The actual singularity. torch's binary_cross_entropy FLOORS log at
            # -100; a kernel without that floor returns -inf here instead. At
            # p=1e-7 the floor never engages (log(1e-7) = -16.1), so the
            # near_zero_prob variant above cannot distinguish them -- measured.
            ("exact_zero_and_one", _pack(
                torch.where(torch.arange(x.numel(), device=x.device).reshape(x.shape) % 2 == 0,
                            torch.zeros_like(x), torch.ones_like(x)))),
        ]


def get_spec() -> BceLossSpec:
    return BceLossSpec(name="bce_loss")
