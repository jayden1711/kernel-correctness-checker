"""KernelSpec for min_reduction  f(x) -> Tensor(n_rows,), reduces last dim."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.reduction_properties import (
    check_permutation_invariance,
    check_shift_equivariance,
    check_positive_scale_equivariance,
)


class MinReductionSpec(SingleTensorSpec):
    name: str = "min_reduction"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("permutation_invariance", check_permutation_invariance),
            ("shift_equivariance", check_shift_equivariance),
            ("positive_scale_equivariance", check_positive_scale_equivariance),
        ]

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs
        return [
            ("all_positive_nonpow2", torch.rand(x.shape[0], 100, device=x.device) + 0.1),
            ("large_magnitude", x * 500),
        ]


def get_spec() -> MinReductionSpec:
    return MinReductionSpec(name="min_reduction")
