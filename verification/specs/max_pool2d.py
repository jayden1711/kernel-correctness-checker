"""KernelSpec for max_pool2d  f(x, kernel_size, stride, padding) -> Tensor."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import PoolKernelSpec
from verification.layer3_properties.pool_properties import (
    check_maxpool_shift_equivariance,
    check_maxpool_positive_scale_equivariance,
)


class MaxPool2dSpec(PoolKernelSpec):
    name: str = "max_pool2d"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("shift_equivariance", lambda cf, inputs: check_maxpool_shift_equivariance(cf, *inputs)),
            ("positive_scale_equivariance",
             lambda cf, inputs: check_maxpool_positive_scale_equivariance(cf, *inputs)),
        ]

    @property
    def valid_shapes(self):
        # (N, C, H, W, kernel_size, stride, padding)
        return [(4, 8, 33, 33, 3, 2, 1), (2, 4, 17, 17, 2, 2, 0), (1, 8, 9, 9, 3, 1, 1)]

    def get_adversarial_inputs(self, inputs):
        x, kernel_size, stride, padding = inputs
        all_negative = -x.abs() - 0.1
        return [("all_negative_padded", (all_negative, kernel_size, stride, max(padding, 1)))]

    def make_inputs(self, shape, device, dtype):
        N, C, H, W, kernel_size, stride, padding = shape
        x = torch.randn(N, C, H, W, device=device, dtype=dtype)
        return x, kernel_size, stride, padding


def get_spec() -> MaxPool2dSpec:
    return MaxPool2dSpec(name="max_pool2d")
