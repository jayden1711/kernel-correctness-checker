"""KernelSpec for max_pool3d  f(x, kernel_size, stride, padding) -> Tensor."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import PoolKernelSpec
from verification.layer3_properties.pool_properties import (
    check_maxpool_shift_equivariance,
    check_maxpool_positive_scale_equivariance,
)


class MaxPool3dSpec(PoolKernelSpec):
    name: str = "max_pool3d"
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
        # (N, C, D, H, W, kernel_size, stride, padding)
        return [(2, 4, 17, 17, 17, 3, 2, 1), (1, 4, 9, 9, 9, 2, 2, 0)]

    def get_adversarial_inputs(self, inputs):
        x, kernel_size, stride, padding = inputs
        all_negative = -x.abs() - 0.1
        return [("all_negative_padded", (all_negative, kernel_size, stride, max(padding, 1)))]

    def make_inputs(self, shape, device, dtype):
        N, C, D, H, W, kernel_size, stride, padding = shape
        x = torch.randn(N, C, D, H, W, device=device, dtype=dtype)
        return x, kernel_size, stride, padding


def get_spec() -> MaxPool3dSpec:
    return MaxPool3dSpec(name="max_pool3d")
