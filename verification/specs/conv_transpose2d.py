"""KernelSpec for conv_transpose2d — f(x, W, stride, padding, dilation, groups) -> Tensor.

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
class ConvTranspose2dSpec(ConvKernelSpec):
    name: str = "conv_transpose2d"
    requires_backward: bool = False
    transposed: bool = True
    depthwise: bool = False


    # DILATION COVERAGE FIX (2026-08-27). Config index 2 originally carried
    # dilation = 1, which meant NO config in this operator's sweep exercised
    # dilation at all -- so a kernel that silently ignores the dilation
    # argument produced bit-identical output on every generated input and was
    # undetectable. Measured: the `ignores_dilation` mutant escaped the whole
    # battery for conv3d until this changed.
    #
    # Same class of defect as Phase 1's D1 (a hyperparameter present in the
    # signature but never varied), found the same way -- by a mutant escaping.
    # conv1d/conv2d/conv_transpose1d already had a dilated config; conv3d,
    # conv_transpose2d, conv_transpose3d and depthwise_conv2d did not.
    # pointwise_conv2d is deliberately exempt: its kernel is 1x1, so dilation
    # is a no-op by definition and varying it would test nothing.
    @property
    def valid_shapes(self):
        # (N, C_in, C_out, spatial, kernel, stride, padding, dilation, groups)
        return [(2, 4, 6, (16, 15), (3, 3), 1, 1, 1, 1), (2, 4, 6, (9, 8), (3, 3), 2, 1, 1, 1), (1, 8, 8, (11, 9), (3, 5), 2, (1, 2), 2, 2), (2, 3, 5, (7, 6), (3, 3), 1, 0, 1, 1), (1, 2, 2, (5, 5), (1, 1), 1, 0, 1, 1)]

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
            # Transposed conv with stride>1 scatters into a periodic pattern; this makes the stride-phase
            # structure explicit, so a kernel with the wrong divisibility test fails visibly.
            ("checkerboard_probe", _pack(torch.zeros_like(x).index_fill_(-1, torch.arange(0, x.shape[-1], 2, device=x.device), 1.0))),
        ]


def get_spec() -> ConvTranspose2dSpec:
    return ConvTranspose2dSpec(name="conv_transpose2d")
