"""KernelSpec for argmax  f(x) -> Tensor(n_rows,) int64, reduces last dim."""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec
from verification.layer3_properties.argextreme_properties import (
    check_shift_invariance,
    check_positive_scale_invariance,
)


class ArgmaxSpec(SingleTensorSpec):
    name: str = "argmax"
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
        """
        FIXED: the original 2-duplicate-position trigger did NOT catch
        the tiebreak mutant in a real run.check_perturbation_tolerance's
        adaptive tolerance absorbed a small index gap (2 tied positions
        out of 16 -> reference index 2 vs mutant index 11, diff=9) as
        within-tolerance. The mutant only got caught via an unrelated
        generic check (check_weight_magnitude's large_uniform variant,
        which ties ALL columns and produces the maximum possible index
        gap, diff=511 on a 512-wide row). Fixed by using a fully-tied
        row here too, so THIS check does the job it's meant to instead
        of relying on an incidental side effect elsewhere.
        """
        x = inputs.clone()
        x[:] = 1.0  # every column tied for the max -- maximizes the
                    # first-occurrence vs last-occurrence index gap
        return [("duplicate_max", x)]


def get_spec() -> ArgmaxSpec:
    return ArgmaxSpec(name="argmax", output_dtype=torch.int64)
