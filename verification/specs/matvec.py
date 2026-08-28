"""KernelSpec for matvec — f(A, v) -> A @ v

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "matvec";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import MatvecKernelSpec


@dataclass
class MatvecSpec(MatvecKernelSpec):
    name: str = "matvec"
    requires_backward: bool = False

    @property
    def valid_shapes(self):
        return [(512, 512), (1024, 256), (1, 512), (333, 129), (2048, 8)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # Accumulator overflow in fp16 reductions.
            ("large_magnitude", _pack(x * 1e4)),
            # First output element must be exactly 0. Catches an uninitialised accumulator, which random data hides because garbage + real sum still looks plausible.
            ("row_of_zeros", _pack(torch.cat([torch.zeros_like(x[:1]), x[1:]], dim=0))),
            # Cancellation in the K reduction.
            ("alternating_signs", _pack(torch.where(torch.arange(x.shape[-1], device=x.device) % 2 == 0, x.abs(), -x.abs()))),
        ]


def get_spec() -> MatvecSpec:
    return MatvecSpec(name="matvec")
