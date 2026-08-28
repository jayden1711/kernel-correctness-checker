"""KernelSpec for cumsum — f(x) -> inclusive prefix sum over last dim

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "cumsum";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec


@dataclass
class CumsumSpec(SingleTensorSpec):
    name: str = "cumsum"
    requires_backward: bool = False

    @property
    def valid_shapes(self):
        return [(64, 512), (8, 1024), (1, 4096), (128, 333), (4, 1)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # Massive cancellation: the running sum stays near zero while |x| does not. Catches a kernel that accumulates in low precision -- the error is invisible on same-sign data.
            ("alternating_signs", _pack(torch.where(torch.arange(x.shape[-1], device=x.device) % 2 == 0, x.abs(), -x.abs()))),
            # One huge leading value swamps every later addition in fp32. A correct scan still tracks the tail; a naive one loses it entirely.
            ("large_then_tiny", _pack(torch.cat([x[..., :1] * 1e8, x[..., 1:] * 1e-8], dim=-1))),
            # Output must be exactly the index ramp -- makes an off-by-one in inclusive/exclusive boundaries a hard, readable failure instead of a small numeric one.
            ("all_ones", _pack(torch.ones_like(x))),
            # Tile-boundary bug in the last partial tile -- the scan's carry across tiles is the single most common Triton scan defect.
            ("non_power_of_two", _pack(torch.randn(x.shape[0], 333, device=x.device, dtype=x.dtype) if x.dim() == 2 else x)),
        ]


def get_spec() -> CumsumSpec:
    return CumsumSpec(name="cumsum")
