"""KernelSpec for cross_entropy  f(logits, targets) -> scalar Tensor."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import CrossEntropyKernelSpec
from verification.layer3_properties.cross_entropy_properties import (
    check_non_negativity,
    check_shift_invariance,
)


class CrossEntropySpec(CrossEntropyKernelSpec):
    name: str = "cross_entropy"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ("non_negativity", _wrap_loss(check_non_negativity)),
            ("shift_invariance", lambda cf, inputs: check_shift_invariance(cf, *inputs)),
        ]

    @property
    def valid_shapes(self):
        return [(64, 100), (32, 1000), (1, 50), (256, 10), (128, 333)]

    def get_adversarial_inputs(self, inputs):
        logits, targets = inputs
        large_magnitude = torch.full_like(logits, 150.0) + logits
        return [("large_magnitude_logits", (large_magnitude, targets))]


def get_spec() -> CrossEntropySpec:
    return CrossEntropySpec(name="cross_entropy")


def _wrap_loss(check_fn):
    def wrapped(candidate_fn, inputs):
        logits, targets = inputs
        loss = candidate_fn(logits, targets)
        return check_fn(loss)
    return wrapped
