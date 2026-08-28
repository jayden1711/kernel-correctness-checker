"""KernelSpec for logsumexp — f(x) -> logsumexp over last dim

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "logsumexp";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import SingleTensorSpec


def _check_ge_rowmax(candidate_fn, inputs):
    """logsumexp(x) >= max(x), always, with equality only as n -> 1.

    Cheap, exact, and it fails loudly for the first-tile class of bug that the
    max_in_last_tile adversarial input targets.
    """
    y = candidate_fn(inputs)
    rm = inputs.float().max(dim=-1).values
    bad = (y.float() < rm - 1e-3).sum().item()
    return (bad == 0), (f"{bad} row(s) below their own max" if bad
                        else "logsumexp >= row max on all rows")


@dataclass
class LogsumexpSpec(SingleTensorSpec):
    name: str = "logsumexp"
    requires_backward: bool = False

    @property
    def algebraic_properties(self):
        return [
            ('ge_rowmax', lambda cf, inputs: _check_ge_rowmax(cf, inputs)),
        ]

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # Without the max-subtraction trick exp() overflows outright. This is logsumexp's entire reason for existing as a fused kernel, and random data never triggers it.
            ("extreme_range", _pack(torch.cat([torch.full_like(x[:, :1], 1e4), x[:, 1:] - 1e4], dim=-1) if x.dim() == 2 else x * 1e4)),
            # Answer is exactly 1 + log(n) -- a hand-checkable constant.
            ("all_equal", _pack(torch.ones_like(x))),
            # The running max arrives in the final tile, so a kernel that fixes its max from the first tile is numerically wrong here and only here.
            ("max_in_last_tile", _pack(torch.zeros_like(x).index_fill_(-1, torch.tensor([x.shape[-1]-1], device=x.device), 1e4))),
        ]


def get_spec() -> LogsumexpSpec:
    return LogsumexpSpec(name="logsumexp")
