"""KernelSpec for avg_pool3d  f(x, kernel_size, stride, padding) -> Tensor."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import PoolKernelSpec
from verification.layer3_properties.pool_properties import (
    check_avgpool_shift_equivariance_no_padding,
    check_avgpool_positive_scale_equivariance,
)


class AvgPool3dSpec(PoolKernelSpec):
    name: str = "avg_pool3d"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("shift_equivariance_no_padding",
             lambda cf, inputs: check_avgpool_shift_equivariance_no_padding(cf, *inputs)),
            ("positive_scale_equivariance",
             lambda cf, inputs: check_avgpool_positive_scale_equivariance(cf, *inputs)),
        ]

    @property
    def valid_shapes(self):
        return [(2, 4, 17, 17, 17, 3, 2, 1), (1, 4, 9, 9, 9, 2, 2, 0)]

    def get_adversarial_inputs(self, inputs):
        x, kernel_size, stride, padding = inputs
        return [("padded", (x, kernel_size, stride, max(padding, 1)))]


def get_spec() -> AvgPool3dSpec:
    return AvgPool3dSpec(name="avg_pool3d")
