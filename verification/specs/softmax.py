from dataclasses import dataclass
from typing import List, Tuple, Callable
import torch

@dataclass  
class KernelSpec:
    name: str
    adversarial_inputs: Callable      # generates targeted failure-mode inputs
    algebraic_properties: List[Tuple[str, Callable]]  # (name, check_fn) pairs
    valid_shapes: List[Tuple]         # shapes to test cross-shape generalization
    requires_backward: bool           # whether to verify gradient correctness

# softmax spec
def get_spec() -> KernelSpec:
    return KernelSpec(
        name="softmax",

        adversarial_inputs=lambda x: [
            ("max_in_last_tile", _max_in_last_tile(x)),
            ("equal_logits",     _equal_logits(x)),
            ("extreme_range",    _extreme_range(x)),
            ("non_power_of_two", _non_power_of_two(x)),
        ],

        algebraic_properties=[
            ("rows_sum_to_one",   check_rows_sum_to_one),
            ("shift_invariance",  check_shift_invariance),
            ("monotonicity",      check_monotonicity),
            ("precision_coercion",check_precision_coercion),
        ],

        valid_shapes=[
            (512, 512),      # base case
            (256, 1024),     # wide
            (1, 512),        # batch size 1
            (1000, 333),     # non-power-of-two
            (2048, 128),     # tall and narrow
        ],

        requires_backward=True,
    )

# adversarial input generators
def _max_in_last_tile(x):
    """Max value sits in last tile — exposes first-tile-only kernels."""
    result = torch.zeros_like(x)
    result[:, -1] = 1e9
    return result

def _equal_logits(x):
    """All logits equal — wrong answer is accidentally close to right answer."""
    return torch.ones_like(x)

def _extreme_range(x):
    """Extreme dynamic range — exposes partial accumulation."""
    result = torch.randn_like(x)
    result[:, 0] = 1e9
    result[:, -1] = -1e9
    return result

def _non_power_of_two(x):
    """Non-power-of-two columns — exposes tile boundary bugs."""
    n_rows = x.shape[0]
    return torch.randn(n_rows, 333, device=x.device, dtype=x.dtype)