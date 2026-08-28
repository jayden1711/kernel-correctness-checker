"""KernelSpec for conv_transpose1d — f(x, W, stride, padding, dilation, groups) -> Tensor.

Added 2026-08-27 (Phase 2). Closed-form ||J_o||_2 lives in
verification/layer2_numeric_oracle/structural_l.py: one identity,
||J_o|| = sqrt(F(ones, W^2)[o]), covers every conv form. Derivation-verified
against autograd (19 configs, max rel err 3.8e-16) and probe-verified natively
on real Triton kernels.
"""

from dataclasses import dataclass
import torch

from verification.specs.base_spec import ConvKernelSpec


@dataclass
class ConvTranspose1dSpec(ConvKernelSpec):
    name: str = "conv_transpose1d"
    requires_backward: bool = False
    transposed: bool = True
    depthwise: bool = False

    @property
    def valid_shapes(self):
        # (N, C_in, C_out, spatial, kernel, stride, padding, dilation, groups)
        return [(2, 4, 6, (32,), (3,), 1, 1, 1, 1), (2, 4, 6, (17,), (3,), 2, 1, 1, 1), (1, 8, 8, (21,), (5,), 2, 2, 2, 2), (2, 3, 5, (11,), (3,), 3, 1, 1, 1), (1, 2, 2, (9,), (1,), 1, 0, 1, 1)]

    def get_adversarial_inputs(self, inputs):
        x = inputs[0]
        rest = tuple(inputs[1:])
        def _pack(t):
            return (t,) + rest
        return [
            # Zeros the spatial border and leaves the interior. Border outputs are exactly the ones whose
            # receptive field is truncated by padding, so a kernel with an off-by-one bound check
            # differs ONLY here -- random data averages the error away across the interior.
            ("zero_border_probe", _pack(torch.nn.functional.pad(x[..., 1:-1].contiguous(), [1,1]*(x.dim()-2)) if x.shape[-1] > 2 else x)),
            # One input channel set to all-ones, the rest zero. The output must be exactly the sum of that
            # channel's taps -- reads the weight layout straight out of the result, so a transposed
            # or mis-grouped weight index is unmistakable rather than merely numerically off.
            ("single_channel_impulse", _pack(torch.zeros_like(x).index_fill_(1, torch.tensor([0], device=x.device), 1.0))),
            # Accumulator overflow in the channel/tap reduction.
            ("large_magnitude", _pack(x * 1e4)),
        ]


def get_spec() -> ConvTranspose1dSpec:
    return ConvTranspose1dSpec(name="conv_transpose1d")
