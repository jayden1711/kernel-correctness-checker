"""
KernelSpec for batchnorm  f(x, running_mean, running_var, weight, bias) -> Tensor.
INFERENCE MODE ONLY.

DELIBERATELY NO algebraic_properties (uses the KernelSpec base class's
default empty list). Inference-mode BatchNorm normalizes using FIXED
EXTERNAL running statistics, not statistics computed from x itself -- so
"output has zero mean" is only true when x's actual distribution happens
to match running_mean/running_var, which adversarial inputs are
specifically constructed NOT to do. Asserting it as an algebraic
property would make the checker reject correct BatchNorm kernels on
exactly the inputs the adversarial search is supposed to explore. Layers
1+2 (structural checks, perturbation tolerance, cross-shape) still fully
apply -- only Layer 3 is intentionally skipped for this operator.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import BatchNormKernelSpec


class BatchNormSpec(BatchNormKernelSpec):
    name: str = "batchnorm"
    requires_backward: bool = False

    @property
    def valid_shapes(self):
        # (N, C, H, W)
        return [(4, 8, 16, 16), (2, 4, 32, 32), (1, 8, 8, 8), (4, 16, 5, 7)]

    def get_adversarial_inputs(self, inputs):
        x, running_mean, running_var, weight, bias = inputs
        # Deliberately mismatched from running_mean/running_var --
        # exactly the case that would make a zero_mean-style property
        # false even for a CORRECT kernel, which is why none is asserted.
        far_from_stats = x * 50 + 100
        return [("far_from_running_stats", (far_from_stats, running_mean, running_var, weight, bias))]


def get_spec() -> BatchNormSpec:
    return BatchNormSpec(name="batchnorm")
