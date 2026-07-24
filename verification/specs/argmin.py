"""KernelSpec for argmin  f(x) -> Tensor(n_rows,) int64, reduces last dim."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.argextreme_properties import (
    check_shift_invariance,
    check_positive_scale_invariance,
)


class ArgminSpec(SingleTensorSpec):
    name: str = "argmin"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("shift_invariance", check_shift_invariance),
            ("positive_scale_invariance", check_positive_scale_invariance),
        ]

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        """FIXED -- see argmax.py's identical fix for the reasoning:
        a fully-tied row maximizes the first-vs-last-occurrence index
        gap instead of relying on an incidental catch elsewhere."""
        x = inputs.clone()
        x[:] = 1.0
        return [("duplicate_min", x)]


def get_spec() -> ArgminSpec:
    return ArgminSpec(name="argmin", output_dtype=torch.int64)
