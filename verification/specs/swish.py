"""KernelSpec for swish  f(x) -> Tensor, x any shape (elementwise)."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.swish_properties import (
    check_zero_at_origin,
    check_monotonic_nonneg,
)


class SwishSpec(SingleTensorSpec):
    name: str = "swish"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("zero_at_origin", lambda cf, inputs: check_zero_at_origin(cf)),
            ("monotonic_nonneg", lambda cf, inputs: check_monotonic_nonneg(cf, inputs)),
        ]

    @property
    def valid_shapes(self):
        return [(4096,), (1024,), (100000,), (1,), (333,)]

    def get_adversarial_inputs(self, inputs):
        x = inputs
        return [
            ("large_magnitude", x * 100),
            ("near_global_min", torch.full_like(x, -1.2785) + x * 0.01),
        ]

    def make_inputs(self, shape, device, dtype):
        return torch.randn(*shape, device=device, dtype=dtype)


def get_spec() -> SwishSpec:
    return SwishSpec(name="swish")
