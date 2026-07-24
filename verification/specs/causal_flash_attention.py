"""
KernelSpec for causal_flash_attention  f(Q, K, V) -> Tensor.
Reuses AttentionKernelSpec unchanged. Same convex_hull_bound property as
non-causal SDPA -- verified true for causal too, see the property
module's docstring for the subset-of-full-min/max argument.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import AttentionKernelSpec
from verification.layer3_properties.attention_convexity_properties import (
    check_convex_hull_bound,
)


class CausalFlashAttentionSpec(AttentionKernelSpec):
    name: str = "causal_flash_attention"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("convex_hull_bound", lambda cf, inputs: check_convex_hull_bound(cf, *inputs)),
        ]

    @property
    def valid_shapes(self):
        return [(128, 64), (256, 32), (64, 128), (1, 64), (333, 64)]

    def get_adversarial_inputs(self, inputs):
        Q, K, V = inputs
        return [("large_magnitude_qk", (Q * 20, K * 20, V))]


def get_spec() -> CausalFlashAttentionSpec:
    return CausalFlashAttentionSpec(name="causal_flash_attention")
