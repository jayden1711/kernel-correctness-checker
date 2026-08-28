"""KernelSpec for kldiv_loss — f(log_q, p) -> scalar KL divergence (batchmean)

Added 2026-08-27 (Phase 1). Closed-form ||J_i||_2 lives in
verification/layer2_numeric_oracle/structural_l.py under op key "kldiv_loss";
derivation-verified against autograd, NOT probe-verified on a real Triton
kernel — see that file's Phase-1 note.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import TargetLossKernelSpec



@dataclass
class KldivLossSpec(TargetLossKernelSpec):
    name: str = "kldiv_loss"
    requires_backward: bool = False

    def make_inputs(self, shape, device, dtype):
        """KL divergence takes LOG-probabilities as input and a probability
        simplex as target. randn for either is out of domain."""
        log_q = torch.log_softmax(
            torch.randn(*shape, device=device, dtype=dtype), dim=-1)
        p = torch.softmax(
            torch.randn(*shape, device=device, dtype=dtype), dim=-1)
        return log_q, p

    @property
    def valid_shapes(self):
        return [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        rest = inputs[1:] if isinstance(inputs, tuple) else ()
        def _pack(t):
            return (t,) + rest if rest else t
        return [
            # KL(p||p) = 0 exactly. The one input where the answer is known in closed form.
            ("identical_distributions", _pack(torch.log(rest[0]) if rest else x)),
            # Target mass near zero exercises the 0*log0 convention.
            ("near_zero_target", _pack(x)),
            # log q is uniform; KL reduces to -H(p) - log(1/n), a hand-checkable value.
            ("uniform_input", _pack(torch.full_like(x, -float(torch.log(torch.tensor(float(x.shape[-1]))))) )),
        ]


def get_spec() -> KldivLossSpec:
    return KldivLossSpec(name="kldiv_loss")
