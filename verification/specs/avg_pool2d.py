"""KernelSpec for avg_pool2d  f(x, kernel_size, stride, padding) -> Tensor."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import PoolKernelSpec
from verification.layer3_properties.pool_properties import (
    check_avgpool_shift_equivariance_no_padding,
    check_avgpool_positive_scale_equivariance,
)


class AvgPool2dSpec(PoolKernelSpec):
    name: str = "avg_pool2d"
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
        return [(4, 8, 33, 33, 3, 2, 1), (2, 4, 17, 17, 2, 2, 0), (1, 8, 9, 9, 3, 1, 1)]

    def get_adversarial_inputs(self, inputs):
        x, kernel_size, stride, padding = inputs
        return [("padded", (x, kernel_size, stride, max(padding, 1)))]


def get_spec() -> AvgPool2dSpec:
    return AvgPool2dSpec(name="avg_pool2d")
