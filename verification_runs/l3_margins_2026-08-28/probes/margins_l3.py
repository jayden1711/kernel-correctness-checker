"""
Margin analysis for the Layer-3 tolerance constants (item 5, 2026-08-28):

  A. the 0.9 precision-coercion factor (matmul / flash_attention / rmsnorm
     / softmax) -- from the banked arm's recorded fp32/fp16 error pairs:
     for every record where the factor is LIVE (err_fp32 > atol and fp16
     ran), the flip point is r = err_fp32/err_fp16; the factor's margin is
     its distance to the nearest r on either side.
  B. the unit_variance atol asymmetry (groupnorm/instancenorm 3e-2 vs
     layernorm 1e-3) and the unit_rms/scale-invariance pass margins --
     banked where print precision allows, CPU fp32 emulation where it
     truncates (layernorm/rmsnorm pass records print no value).
  C. the non-round probe constants (c = 4.2 l1/l2/frobenius, 3.1
     instancenorm, 2.9 groupnorm): every mutant these checks see is
     exactly 1-homogeneous (normalizations x/S(x) with S 1-homogeneous),
     so the checks are STRUCTURALLY INERT in c -- verified by sweeping c
     and recording the max deviation for reference and mutant arithmetic.

Run:  .venv/bin/python margins_l3.py
"""
import gzip
import json
import os
import re

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.path.join(HERE, "..", "..", "layernorm_mask_fix_2026-08-28",
                   "arms", "G_lnfix.json.gz")
torch.manual_seed(20260828)

# ---------------------------------------------------------------- A ------
print("A. the 0.9 precision-coercion factor")
d = json.load(gzip.open(ARM, "rt"))
pair = re.compile(r"fp32_err=(\d+\.\d+), fp16_err=(\d+\.\d+)")
ATOL_PC = 1e-3
live, dead_atol = [], []
for e in d["entries"]:
    packs = [("mutant", e["mutant"]["records"])] + \
            [(f"ref{i}", r["records"]) for i, r in enumerate(e["refs"])]
    for tag, recs in packs:
        for r in recs:
            if r["name"] != "precision_coercion":
                continue
            m = pair.search(r.get("detail") or "")
            if not m:
                continue    # layernorm (no factor), flash (fp16 unsupported)
            e32, e16 = float(m.group(1)), float(m.group(2))
            rec = (e["op"], e["mutant"]["name"], tag, e32, e16, r["outcome"])
            (live if e32 > ATOL_PC else dead_atol).append(rec)
print(f"  {len(live)} live records (err_fp32 > atol), "
      f"{len(dead_atol)} atol-gated (factor irrelevant)")
rs = sorted((e32 / e16, op, mu, tag, out) for op, mu, tag, e32, e16, out in live)
for r_, op, mu, tag, out in rs:
    print(f"    r = {r_:.5f}  {op}/{mu}/{tag}  ({out})")
above = [r_ for r_, *_ in rs if r_ > 0.9]
below = [r_ for r_, *_ in rs if r_ < 0.9]
print(f"  dead zone around 0.9: ({max(below) if below else 0:.4f}, "
      f"{min(above) if above else 'inf':.4f}) -- factor can move to either "
      f"edge before any check-level outcome flips")
# the atol arm of the same check
e32s = sorted(set(e32 for _, _, _, e32, _, _ in live + dead_atol))
just_above = min((x for x in e32s if x > ATOL_PC), default=None)
just_below = max((x for x in e32s if x <= ATOL_PC), default=None)
print(f"  pc atol=1e-3 arm: nearest err_fp32 above = {just_above} "
      f"({just_above/ATOL_PC:.1f}x), nearest below = {just_below} "
      f"({ATOL_PC/just_below if just_below else float('inf'):.0f}x)")

# ---------------------------------------------------------------- B ------
print("\nB. unit-variance / unit-rms / scale-invariance margins")
print("  banked (print precision 1e-6):")
print("    groupnorm uv devs: mutant 1.4e-5, refs ~1.3e-5  vs atol 3e-2 "
      "-> margin ~2.1e3x; would ALSO pass layernorm's 1e-3 (71x)")
print("    instancenorm uv devs: <=1.6e-5 vs 3e-2 -> ~1.9e3x; passes 1e-3 too")

N, DCOL = 64, 128


def ln_ref(x, g, b):
    m = x.mean(-1, keepdim=True)
    v = ((x - m) ** 2).mean(-1, keepdim=True)
    return (x - m) / torch.sqrt(v + 1e-5) * g + b


def ln_wrong_var(x, g, b):
    m = x.mean(-1, keepdim=True)
    v = (x * x).mean(-1, keepdim=True) - m * m
    return (x - m) / torch.sqrt(v + 1e-5) * g + b


def ln_skip_mean(x, g, b):
    m = x.mean(-1, keepdim=True)
    v = ((x - m) ** 2).mean(-1, keepdim=True)
    return x / torch.sqrt(v + 1e-5) * g + b


def ln_ignore_gb(x, g, b):
    m = x.mean(-1, keepdim=True)
    v = ((x - m) ** 2).mean(-1, keepdim=True)
    return (x - m) / torch.sqrt(v + 1e-5)


def rms_ref(x, g):
    r = torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)
    return x / r * g


def rms_ignore_gamma(x, g):
    r = torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)
    return x / r


def uv_dev(out):
    return float((out.var(-1, unbiased=False) - 1).abs().max())


def rms_dev(out):
    return float((torch.sqrt((out * out).mean(-1)) - 1).abs().max())


def si_dev(fn, args, c=100.0):
    return float((fn(*args) - fn(args[0] * c, *args[1:])).abs().max())


devs = {"ln uv ref": [], "ln uv wrong_var": [], "ln uv skip_mean": [],
        "ln uv ignore_gb": [], "ln si ref": [], "rms rms ref": [],
        "rms rms ignore_gamma": [], "rms si ref": []}
for s in range(10):
    torch.manual_seed(s)
    x = torch.randn(N, DCOL)
    g = torch.ones(DCOL)
    b = torch.zeros(DCOL)
    devs["ln uv ref"].append(uv_dev(ln_ref(x, g, b)))
    devs["ln uv wrong_var"].append(uv_dev(ln_wrong_var(x, g, b)))
    devs["ln uv skip_mean"].append(uv_dev(ln_skip_mean(x, g, b)))
    devs["ln uv ignore_gb"].append(uv_dev(ln_ignore_gb(x, g, b)))
    devs["ln si ref"].append(si_dev(ln_ref, (x, g, b)))
    devs["rms rms ref"].append(rms_dev(rms_ref(x, g)))
    devs["rms rms ignore_gamma"].append(rms_dev(rms_ignore_gamma(x, g)))
    devs["rms si ref"].append(si_dev(rms_ref, (x, g)))
print("  emulated (10 seeds, corpus (64,128), gamma=1/beta=0; atol 1e-3):")
for k, v in devs.items():
    worst = max(v)
    print(f"    {k:22s} worst dev = {worst:.3e}   margin to atol 1e-3 = "
          f"{1e-3/worst if worst > 0 else float('inf'):.0f}x")

# ---------------------------------------------------------------- C ------
print("\nC. probe-constant sweeps: 1-homogeneous mutants => inert in c")


def l1_ref(x):
    return x / (x.abs().sum(-1, keepdim=True) + 1e-12)


def l1_partial(x):
    h = x.shape[-1] // 2
    return x / (x[..., :h].abs().sum(-1, keepdim=True) + 1e-12)


def l2_ref(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-12)


def l2_wrong(x):
    return x / (x.abs().sum(-1, keepdim=True) + 1e-12)


def fro_ref(x):
    return x / (torch.sqrt((x * x).sum()) + 1e-12)


def fro_wrong(x):
    return x / (x.abs().sum() + 1e-12)


def inst_skip_eps(x):        # (N,C,L): normalize over trailing dims, no eps
    m = x.mean(dim=-1, keepdim=True)
    v = ((x - m) ** 2).mean(dim=-1, keepdim=True)
    return (x - m) / torch.sqrt(v)


ATOL_SI = 1e-3
for name, fn, shape in (("l1/ref", l1_ref, (N, DCOL)),
                        ("l1/partial_reduction", l1_partial, (N, DCOL)),
                        ("l2/ref", l2_ref, (N, DCOL)),
                        ("l2/wrong_norm", l2_wrong, (N, DCOL)),
                        ("frob/ref", fro_ref, (N, DCOL)),
                        ("frob/wrong_norm", fro_wrong, (N, DCOL)),
                        ("inst/skip_eps", inst_skip_eps, (4, 8, 16))):
    torch.manual_seed(1)
    x = torch.randn(*shape)
    row = []
    for c in (1.3, 2.9, 3.1, 4.2, 10.0, 100.0, 1000.0):
        dev = float((fn(x) - fn(x * c)).abs().max())
        row.append(f"c={c:g}: {dev:.1e}")
    print(f"  {name:22s} " + "  ".join(row))
print("  (atol 1e-3: every entry above is >=1e4x inside; no c in [1.3, 1e3]"
      " flips any outcome)")
