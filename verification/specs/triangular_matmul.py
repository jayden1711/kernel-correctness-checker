"""KernelSpec for triangular_matmul — f(A, B) -> tril(A @ B)

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "triangular_matmul";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import TriangularMatmulKernelSpec


@dataclass
class TriangularMatmulSpec(TriangularMatmulKernelSpec):
    name: str = "triangular_matmul"
    requires_backward: bool = False

    @property
    def valid_shapes(self):
        return [(512,), (256,), (64,), (333,), (1,)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # Feeds a strictly-upper A. The masked output region must still be exactly zero -- catches a kernel that computes the full product and forgets the mask.
            ("upper_only_input", _pack(torch.triu(x, diagonal=1))),
            # Output must be tril(B) exactly.
            ("identity", _pack(torch.eye(x.shape[0], device=x.device, dtype=x.dtype))),
            # Accumulator overflow.
            ("large_magnitude", _pack(x * 1e4)),
        ]


def get_spec() -> TriangularMatmulSpec:
    return TriangularMatmulSpec(name="triangular_matmul")
