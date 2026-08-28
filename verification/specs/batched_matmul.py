"""KernelSpec for batched_matmul — f(A, B) -> torch.bmm(A, B)

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "batched_matmul";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import BatchedMatmulKernelSpec


@dataclass
class BatchedMatmulSpec(BatchedMatmulKernelSpec):
    name: str = "batched_matmul"
    requires_backward: bool = False

    @property
    def valid_shapes(self):
        return [(4, 128, 128, 128), (2, 256, 64, 128), (1, 64, 64, 64), (3, 129, 33, 65), (8, 32, 32, 32)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # fp16 accumulator overflow.
            ("large_magnitude", _pack(x * 1e4)),
            # Every batch but the first must produce exactly 0. Catches a kernel that ignores the batch stride and reuses batch 0's tile for all batches -- the signature bmm bug, and completely invisible on random data where every batch looks equally plausible.
            ("single_batch_nonzero", _pack(torch.cat([x[:1], torch.zeros_like(x[1:])], dim=0))),
            # K-dimension cancellation.
            ("alternating_signs", _pack(torch.where(torch.arange(x.shape[-1], device=x.device) % 2 == 0, x.abs(), -x.abs()))),
        ]


def get_spec() -> BatchedMatmulSpec:
    return BatchedMatmulSpec(name="batched_matmul")
