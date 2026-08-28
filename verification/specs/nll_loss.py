"""KernelSpec for nll_loss — f(log_probs, targets) -> scalar negative log-likelihood

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "nll_loss";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import CrossEntropyKernelSpec


@dataclass
class NllLossSpec(CrossEntropyKernelSpec):
    name: str = "nll_loss"
    requires_backward: bool = False

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # All mass on the correct class: loss ~ 0. Catches a kernel that gathers the wrong index, which random logits make look merely 'a bit high' rather than wrong.
            ("confident_correct", _pack(torch.full_like(x, -20.0).scatter_(1, rest[0].unsqueeze(1), 0.0) if rest else x)),
            # All mass on a wrong class: loss is large and specific.
            ("confident_wrong", _pack(torch.full_like(x, 0.0).scatter_(1, rest[0].unsqueeze(1), -20.0) if rest else x)),
            # Loss must be exactly log(n_classes).
            ("uniform", _pack(torch.full_like(x, -float(torch.log(torch.tensor(float(x.shape[-1]))))))),
        ]


def get_spec() -> NllLossSpec:
    return NllLossSpec(name="nll_loss")
