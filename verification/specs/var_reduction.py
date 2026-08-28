"""KernelSpec for var_reduction — f(x) -> variance over last dim (unbiased)

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "var_reduction";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec


@dataclass
class VarReductionSpec(SingleTensorSpec):
    name: str = "var_reduction"
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
            # Variance must be exactly 0.
            ("constant_rows", _pack(torch.ones_like(x))),
            # Shift-invariance; one-pass cancellation.
            ("large_offset", _pack(x + 1e6)),
            # Dominated by one value.
            ("single_outlier", _pack(torch.zeros_like(x).index_fill_(-1, torch.tensor([0], device=x.device), 1e3))),
        ]


def get_spec() -> VarReductionSpec:
    return VarReductionSpec(name="var_reduction")
