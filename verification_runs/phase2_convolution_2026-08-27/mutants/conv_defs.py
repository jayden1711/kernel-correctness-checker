"""Correct + mutant conv implementations at the TORCH level.

Same rationale as Phase 1's mutants/defs.py: these express real conv failure
modes in a form that runs without a GPU, so the question "do the Phase-2 specs
separate a correct kernel from a buggy one?" can be answered now. Porting a
pinned mutant to Triton is mechanical; guessing whether an unrunnable one is
caught is not.

Every mutant is a documented real conv bug, not a random perturbation.
"""
import torch
import torch.nn.functional as F

OPS = {}
def reg(k, ref, correct, muts): OPS[k] = dict(ref=ref, correct=correct, mutants=muts)

def _flip(W):
    return torch.flip(W, dims=list(range(2, W.dim())))

# --- forward forms ---------------------------------------------------------
for nd, fwd in ((1, F.conv1d), (2, F.conv2d), (3, F.conv3d)):
    def mk(fwd=fwd, nd=nd):
        ref = lambda x, W, s, p, d, g: fwd(x, W, None, s, p, d, g)
        return ref, ref, {
            # correlation vs true convolution: the single most common conv bug,
            # and INVISIBLE for symmetric kernels or symmetric data
            "flipped_kernel": lambda x, W, s, p, d, g, f=fwd: f(x, _flip(W), None, s, p, d, g),
            # ignores dilation -- identical output whenever d == 1, so only the
            # dilated configs in valid_shapes can catch it
            "ignores_dilation": lambda x, W, s, p, d, g, f=fwd: f(x, W, None, s, p, 1, g),
        }
    r, c, m = mk()
    reg(f"conv{nd}d", r, c, m)

# --- transposed forms ------------------------------------------------------
for nd, rev in ((1, F.conv_transpose1d), (2, F.conv_transpose2d), (3, F.conv_transpose3d)):
    def mkT(rev=rev, nd=nd):
        ref = lambda x, W, s, p, d, g: rev(x, W, None, s, p, 0, g, d)
        return ref, ref, {
            "flipped_kernel": lambda x, W, s, p, d, g, f=rev: f(x, _flip(W), None, s, p, 0, g, d),
            # drops the output-padding/stride phase by one: a real
            # divisibility-test off-by-one in the gather formulation
            "wrong_padding": lambda x, W, s, p, d, g, f=rev: f(x, W, None, s, max(0, p - 1), 0, g, d),
        }
    r, c, m = mkT()
    reg(f"conv_transpose{nd}d", r, c, m)

# --- depthwise / pointwise -------------------------------------------------
reg("depthwise_conv2d",
    lambda x, W, s, p, d: F.conv2d(x, W, None, s, p, d, groups=x.shape[1]),
    lambda x, W, s, p, d: F.conv2d(x, W, None, s, p, d, groups=x.shape[1]),
    {
     # not actually grouped -- leaks across channels. Requires W broadcast to a
     # dense filter, which is exactly what a kernel forgetting `groups` does.
     "not_grouped": lambda x, W, s, p, d: F.conv2d(
         x, W.expand(W.shape[0], x.shape[1], *W.shape[2:]).contiguous(), None, s, p, d, 1)
         if W.shape[1] == 1 else F.conv2d(x, W, None, s, p, d, 1),
     "flipped_kernel": lambda x, W, s, p, d: F.conv2d(
         x, _flip(W), None, s, p, d, groups=x.shape[1]),
    })

reg("pointwise_conv2d",
    lambda x, W: F.conv2d(x, W, None, 1, 0, 1, 1),
    lambda x, W: F.conv2d(x, W, None, 1, 0, 1, 1),
    {
     # accumulates only the first half of the input channels
     "partial_channels": lambda x, W: F.conv2d(
         x[:, :max(1, x.shape[1]//2)], W[:, :max(1, W.shape[1]//2)], None, 1, 0, 1, 1),
     # transposes the (C_out, C_in) weight matrix -- valid only when square
     "transposed_weight": lambda x, W: F.conv2d(
         x, W.transpose(0, 1).contiguous(), None, 1, 0, 1, 1)
         if W.shape[0] == W.shape[1] else F.conv2d(x, W * 1.05, None, 1, 0, 1, 1),
    })
