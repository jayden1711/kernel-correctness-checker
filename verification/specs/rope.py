"""KernelSpec for rope — f(x, cos, sin) -> rotary position embedding

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "rope";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import RopeKernelSpec


def _check_norm_preserved(candidate_fn, inputs):
    """RoPE is an orthogonal transform: ||y|| == ||x|| per row, exactly.

    This is the algebraic statement of the same fact the closed-form row norm
    encodes (||J_i|| == 1), so a kernel that fails this will also have a wrong
    Jacobian -- checking it at Layer 2 catches it before the numeric layer pays
    for a full perturbation battery.
    """
    x, cos, sin = inputs
    y = candidate_fn(x, cos, sin)
    nx = x.float().norm(dim=-1)
    ny = y.float().norm(dim=-1)
    bad = ((nx - ny).abs() > 1e-3 * nx.clamp_min(1e-6)).sum().item()
    return (bad == 0), (f"{bad}/{nx.numel()} row(s) changed norm under rotation"
                        if bad else "row norms preserved (orthogonal)")


@dataclass
class RopeSpec(RopeKernelSpec):
    name: str = "rope"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ('norm_preserved', lambda cf, inputs: _check_norm_preserved(cf, inputs)),
        ]

    @property
    def valid_shapes(self):
        return [(512, 128), (1024, 64), (1, 256), (333, 32), (2048, 8)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # RoPE is a rotation, so ||y|| == ||x|| exactly at any scale. A kernel that mixes the wrong halves changes the norm and this makes it a large, obvious error.
            ("norm_preservation_probe", _pack(x * 1e4)),
            # Feeds (1, 0) into every pair: output is exactly (cos, sin). Reads the rotation table straight out of the output -- any half-swap or sign error is unmistakable.
            ("single_axis", _pack(torch.cat([torch.ones_like(x[..., :x.shape[-1]//2]), torch.zeros_like(x[..., x.shape[-1]//2:])], dim=-1))),
            # Interleaved vs split-half pairing conventions disagree here and nowhere else.
            ("alternating_signs", _pack(torch.where(torch.arange(x.shape[-1], device=x.device) % 2 == 0, x.abs(), -x.abs()))),
        ]


def get_spec() -> RopeSpec:
    return RopeSpec(name="rope")
