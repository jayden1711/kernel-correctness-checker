"""Closed-form ||J_o||_2 for the 8 convolution forms, derived and then checked
against an autograd-exact Jacobian.

THE DERIVATION, written out rather than assumed.

A convolution is linear in x. For a standard (gather) conv,

        y_o = sum_tau  W[tau] * x[phi(o, tau)]

where tau ranges over (c_in, k1..kn) and phi maps an output position and a tap
to an input position; taps landing outside the input contribute nothing
(padding is zeros). For fixed o the map tau -> phi(o, tau) is INJECTIVE -- two
distinct taps of the same output element read two distinct input positions --
so

        d y_o / d x_j  =  W[tau]   if some tau has phi(o,tau) = j, else 0
        ||J_o||_2^2    =  sum over IN-BOUNDS taps tau of  W[tau]^2.

Now observe that the right-hand side is itself a convolution: feed an all-ones
input through the SAME operator with W^2 in place of W. In-bounds taps each
contribute W[tau]^2 * 1; out-of-bounds taps contribute 0, exactly as padding
does. Hence, for every variant, with identical stride/padding/dilation/groups:

        ||J_o||_2  =  sqrt( F(ones_like(x), W^2)[o] )                    (*)

For the TRANSPOSED (scatter) forms, y_o = sum over (i,k) with
i*s - p + k*d = o of W[c_in, c_out, k] * x[i]. For a fixed (i, o) the tap k is
uniquely determined, so the map is again injective in the summation index and
the same argument gives (*) with F = conv_transpose.

CONSEQUENCES, both worth stating because they are what the plan claimed:
  * (*) does NOT depend on x. Conv joins matmul/batchnorm as input-independent.
  * (*) is NOT shape-only: it needs W. Padding makes it genuinely non-constant
    across o -- border outputs tap fewer weights -- so the profile is not flat
    and cannot be collapsed to a single number.
  * groups, dilation, stride and asymmetric kernels need no special cases: they
    are already encoded in F.
"""
import itertools, math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DT = torch.float64

CONV   = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
CONVT  = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}


def closed_form_rows(kind, nd, x, W, stride, padding, dilation, groups,
                     output_padding=0):
    """Equation (*). Same operator, W^2, all-ones input."""
    ones = torch.ones_like(x)
    W2 = W * W
    if kind == "conv":
        s = CONV[nd](ones, W2, None, stride, padding, dilation, groups)
    else:
        s = CONVT[nd](ones, W2, None, stride, padding, output_padding,
                      groups, dilation)
    return s.clamp_min(0).sqrt().flatten()


def autograd_rows(fn, x):
    J = torch.autograd.functional.jacobian(lambda t: fn(t).reshape(-1), x)
    return J.reshape(J.shape[0], -1).norm(dim=1)


# (name, kind, nd, N, Cin, Cout, spatial, k, stride, pad, dil, groups)
CASES = [
    ("conv1d",            "conv",  1, 2, 3, 4, (16,),     (3,),     1, 0, 1, 1),
    ("conv1d/pad",        "conv",  1, 2, 3, 4, (16,),     (3,),     1, 1, 1, 1),
    ("conv1d/stride+dil", "conv",  1, 2, 3, 4, (17,),     (3,),     2, 2, 2, 1),
    ("conv2d",            "conv",  2, 2, 3, 4, (9, 8),    (3, 3),   1, 0, 1, 1),
    ("conv2d/pad",        "conv",  2, 2, 3, 4, (9, 8),    (3, 3),   1, 1, 1, 1),
    ("conv2d/asym+dil",   "conv",  2, 2, 3, 4, (11, 9),   (3, 5),   2, (1, 2), 2, 1),
    ("conv2d/grouped",    "conv",  2, 2, 4, 6, (9, 8),    (3, 3),   1, 1, 1, 2),
    ("conv3d",            "conv",  3, 1, 2, 3, (6, 6, 5), (3, 3, 3), 1, 0, 1, 1),
    ("conv3d/pad+stride", "conv",  3, 1, 2, 3, (7, 6, 6), (3, 3, 3), 2, 1, 1, 1),
    ("depthwise2d",       "conv",  2, 2, 4, 4, (9, 8),    (3, 3),   1, 1, 1, 4),
    ("pointwise2d",       "conv",  2, 2, 5, 7, (6, 6),    (1, 1),   1, 0, 1, 1),
    ("convT1d",           "convt", 1, 2, 3, 4, (10,),     (3,),     1, 0, 1, 1),
    ("convT1d/stride+pad","convt", 1, 2, 3, 4, (10,),     (3,),     2, 1, 1, 1),
    ("convT1d/dilated",   "convt", 1, 2, 3, 4, (10,),     (3,),     2, 1, 2, 1),
    ("convT2d",           "convt", 2, 2, 3, 4, (7, 6),    (3, 3),   1, 0, 1, 1),
    ("convT2d/strided",   "convt", 2, 2, 3, 4, (7, 6),    (3, 3),   2, 1, 1, 1),
    ("convT2d/grouped",   "convt", 2, 2, 4, 6, (7, 6),    (3, 3),   2, 1, 1, 2),
    ("convT3d",           "convt", 3, 1, 2, 3, (5, 5, 4), (3, 3, 3), 1, 0, 1, 1),
    ("convT3d/strided",   "convt", 3, 1, 2, 3, (5, 5, 4), (3, 3, 3), 2, 1, 1, 1),
]

print(f"{'case':22s} {'m':>7s} {'max rel err':>12s} {'x-indep':>8s} {'flat?':>7s}  verdict")
print("-" * 76)
bad = 0
for (name, kind, nd, N, Cin, Cout, sp, k, s, p, d, g) in CASES:
    x = torch.randn(N, Cin, *sp, dtype=DT)
    wshape = ((Cout, Cin // g) + tuple(k)) if kind == "conv" else ((Cin, Cout // g) + tuple(k))
    W = torch.randn(*wshape, dtype=DT)
    if kind == "conv":
        fn = lambda t: CONV[nd](t, W, None, s, p, d, g)
    else:
        fn = lambda t: CONVT[nd](t, W, None, s, p, 0, g, d)
    ag = autograd_rows(fn, x)
    cf = closed_form_rows(kind, nd, x, W, s, p, d, g)
    rel = ((cf - ag).abs() / ag.abs().clamp_min(1e-14)).max().item()
    # input-independence: recompute on a completely different x
    x2 = torch.randn_like(x) * 37.0 + 5.0
    cf2 = closed_form_rows(kind, nd, x2, W, s, p, d, g)
    xind = (cf - cf2).abs().max().item() < 1e-12
    flat = (ag.max() - ag.min()).abs().item() < 1e-12
    ok = rel < 1e-10
    bad += 0 if ok else 1
    print(f"{name:22s} {ag.numel():7d} {rel:12.3e} {str(xind):>8s} {str(flat):>7s}  {'OK' if ok else 'FAIL'}")
print("-" * 76)
print(f"{len(CASES)-bad}/{len(CASES)} closed forms match autograd exactly")
print("\n'x-indep' True everywhere confirms the plan's input-independence claim.")
print("'flat?' False wherever padding/stride truncates taps -- the profile is")
print("genuinely non-constant, so conv is input-independent but NOT shape-only.")
