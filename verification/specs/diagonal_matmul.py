"""KernelSpec for diagonal_matmul — f(d, B) -> diag(d) @ B

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "diagonal_matmul";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import DiagonalMatmulKernelSpec


@dataclass
class DiagonalMatmulSpec(DiagonalMatmulKernelSpec):
    name: str = "diagonal_matmul"
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
            # Output must be B's first row and zeros elsewhere. A kernel that materialises the full diagonal matrix and does a dense matmul still passes numerically but this makes any indexing error unmistakable.
            ("one_hot_diagonal", _pack(torch.zeros_like(x).index_fill_(0, torch.tensor([0], device=x.device), 1.0))),
            # Output must be identically zero.
            ("zero_diagonal", _pack(torch.zeros_like(x))),
            # Scaling must be exact -- no reduction involved.
            ("large_magnitude", _pack(x * 1e4)),
        ]


def get_spec() -> DiagonalMatmulSpec:
    return DiagonalMatmulSpec(name="diagonal_matmul")
