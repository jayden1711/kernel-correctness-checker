"""Every Phase-2 conv kernel vs its torch reference, across the full
hyperparameter matrix. Runs BEFORE any measurement -- same gate as Phase 1.

The matrix is the point: one kernel body per form has to cover asymmetric
kernels, stride, padding, dilation and groups, so each is exercised on all of
them rather than on a single default configuration.
"""
import sys, itertools
sys.path.insert(0, "/content")
import torch, torch.nn.functional as F
from conv_kernels import KERNELS

torch.manual_seed(0)
D, DT = "cuda", torch.float32

CASES = []
# (label, op, x-shape, w-shape, kwargs)
for s, p, d, g in [(1,0,1,1), (1,1,1,1), (2,1,1,1), (2,2,2,1), (1,1,1,2), (3,1,2,2)]:
    CASES.append((f"conv1d s{s}p{p}d{d}g{g}", "conv1d", (2,4,33), (6,4//g,3),
                  dict(stride=s,padding=p,dilation=d,groups=g)))
    CASES.append((f"convT1d s{s}p{p}d{d}g{g}", "conv_transpose1d", (2,4,17), (4,6//g,3),
                  dict(stride=s,padding=p,dilation=d,groups=g)))
for s, p, d, g in [(1,0,1,1), (1,1,1,1), (2,1,1,1), (2,2,2,1), (1,1,1,2)]:
    CASES.append((f"conv2d s{s}p{p}d{d}g{g}", "conv2d", (2,4,17,15), (6,4//g,3,3),
                  dict(stride=s,padding=p,dilation=d,groups=g)))
    CASES.append((f"convT2d s{s}p{p}d{d}g{g}", "conv_transpose2d", (2,4,9,8), (4,6//g,3,3),
                  dict(stride=s,padding=p,dilation=d,groups=g)))
# asymmetric kernels / asymmetric hyperparameters
CASES += [
 ("conv2d asym-k",   "conv2d", (2,3,17,15), (5,3,3,5), dict(stride=1,padding=(1,2),dilation=1,groups=1)),
 ("conv2d asym-sd",  "conv2d", (2,3,19,17), (5,3,3,5), dict(stride=(2,1),padding=(1,2),dilation=(2,1),groups=1)),
 ("convT2d asym-k",  "conv_transpose2d", (2,3,9,8), (3,5,3,5), dict(stride=(2,1),padding=(1,2),dilation=1,groups=1)),
 ("conv3d",          "conv3d", (1,3,9,8,7), (4,3,3,3,3), dict(stride=1,padding=1,dilation=1,groups=1)),
 ("conv3d s2p1",     "conv3d", (1,3,9,8,7), (4,3,3,3,3), dict(stride=2,padding=1,dilation=1,groups=1)),
 ("conv3d asym",     "conv3d", (1,2,9,8,7), (4,2,3,1,3), dict(stride=(2,1,1),padding=(1,0,1),dilation=1,groups=1)),
 ("conv3d grouped",  "conv3d", (1,4,8,7,6), (6,2,3,3,3), dict(stride=1,padding=1,dilation=1,groups=2)),
 ("convT3d",         "conv_transpose3d", (1,3,6,5,5), (3,4,3,3,3), dict(stride=1,padding=0,dilation=1,groups=1)),
 ("convT3d s2p1",    "conv_transpose3d", (1,3,6,5,5), (3,4,3,3,3), dict(stride=2,padding=1,dilation=1,groups=1)),
 ("convT3d grouped", "conv_transpose3d", (1,4,5,5,4), (4,3,3,3,3), dict(stride=2,padding=1,dilation=1,groups=2)),
 ("depthwise2d",     "depthwise_conv2d", (2,8,17,15), (8,1,3,3), dict(stride=1,padding=1,dilation=1)),
 ("depthwise2d s2",  "depthwise_conv2d", (2,8,17,15), (8,1,3,3), dict(stride=2,padding=1,dilation=1)),
 ("depthwise2d asym","depthwise_conv2d", (2,8,17,15), (8,1,3,5), dict(stride=1,padding=(1,2),dilation=1)),
 ("pointwise2d",     "pointwise_conv2d", (2,6,12,11), (9,6,1,1), dict()),
]

TREF = {"conv1d":F.conv1d, "conv2d":F.conv2d, "conv3d":F.conv3d,
        "conv_transpose1d":F.conv_transpose1d,
        "conv_transpose2d":F.conv_transpose2d,
        "conv_transpose3d":F.conv_transpose3d}

print(f"{'case':24s} {'out shape':>22s} {'max_err':>11s} {'rel':>10s}  verdict")
print("-"*76)
bad=[]; worst=0.0; worst_case=None
for label, op, xs, ws, kw in CASES:
    try:
        x = torch.randn(*xs, device=D, dtype=DT)
        W = torch.randn(*ws, device=D, dtype=DT)
        got = KERNELS[op](x, W, **kw)
        if op == "depthwise_conv2d":
            exp = F.conv2d(x, W, None, kw.get("stride",1), kw.get("padding",1),
                           kw.get("dilation",1), groups=xs[1])
        elif op == "pointwise_conv2d":
            exp = F.conv2d(x, W, None, 1, 0, 1, 1)
        elif op.startswith("conv_transpose"):
            exp = TREF[op](x, W, None, kw.get("stride",1), kw.get("padding",0),
                           0, kw.get("groups",1), kw.get("dilation",1))
        else:
            exp = TREF[op](x, W, None, kw.get("stride",1), kw.get("padding",0),
                           kw.get("dilation",1), kw.get("groups",1))
        if got.shape != exp.shape:
            print(f"{label:24s} {'--':>22s} {'SHAPE':>11s} {'--':>10s}  FAIL {tuple(got.shape)} vs {tuple(exp.shape)}")
            bad.append(label); continue
        err=(got-exp).abs().max().item(); den=exp.abs().max().item()
        rel=err/den if den>0 else err
        if rel>worst: worst, worst_case = rel, label
        ok = rel < 2e-5
        print(f"{label:24s} {str(tuple(exp.shape)):>22s} {err:11.3e} {rel:10.2e}  {'OK' if ok else 'MISMATCH'}")
        if not ok: bad.append(label)
    except Exception as e:
        print(f"{label:24s} {'--':>22s} {'ERR':>11s} {'--':>10s}  {type(e).__name__}: {str(e)[:50]}")
        bad.append(label)
print("-"*76)
print(f"{len(CASES)-len(bad)}/{len(CASES)} configurations correct")
print(f"WORST relative error: {worst:.3e}  ({worst_case})")
if bad: print("FAILING:", bad)
