"""KernelSpec for mean_reduction  f(x) -> Tensor(n_rows,), reduces last dim."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.reduction_properties import (
    check_permutation_invariance,
    check_scale_linearity,
)


class MeanReductionSpec(SingleTensorSpec):
    name: str = "mean_reduction"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("permutation_invariance", check_permutation_invariance),
            ("scale_linearity", check_scale_linearity),
        ]

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs
        return [
            ("second_half_dominant", torch.cat([torch.zeros_like(x[:, :x.shape[1]//2]),
                                                  x[:, x.shape[1]//2:] * 10], dim=1)),
            ("large_magnitude", x * 500),
        ]


def get_spec() -> MeanReductionSpec:
    return MeanReductionSpec(name="mean_reduction")
