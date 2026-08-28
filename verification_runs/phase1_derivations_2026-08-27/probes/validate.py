"""Phase-1 derivation validation, banked.

Two questions, both device-independent and therefore legitimately answerable on
CPU:

  Q1 CALCULUS. Does each closed form equal the exact Jacobian row norm of the
     torch reference? Checked over 5 seeds and 3 input regimes.

  Q2 PROFILE DEGENERACY. CORPUS_EXPANSION_PLAN.md 4.1 flagged that the
     piecewise-saturating activations may have a row-norm profile that is
     mostly zeros, which would make M3's max-of-|z| simulation degenerate.
     That was a prediction. This measures it: zero-fraction and max/median
     spread of the profile.

NOT ANSWERED HERE, DELIBERATELY: the M3 R^2 over the enlarged operator set.
That needs GPU-measured adaptive_tol from real Triton kernels, and
SESSION_HANDOFF.md 0 forbids substituting a CPU approximation for it.
"""
import json, math, os, sys, statistics as st
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derive import CLOSED, REFS, autograd_row_norms, DT

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
# Input regimes. 'saturating' is the one that answers Q2 -- it is where the
# piecewise activations spend most of their mass in a dead zone.
REGIMES = {"normal": 1.0, "wide": 8.0, "saturating": 40.0}

# Operators whose input must stay in a restricted domain -- scaling their input
# would leave it, so they run in 'normal' only and say so.
DOMAIN_LOCKED = {"bce_loss", "kldiv_loss", "nll_loss", "masked_cumsum"}

rows = []
for op, (fn, mk) in REFS.items():
    for regime, scale in REGIMES.items():
        if op in DOMAIN_LOCKED and regime != "normal":
            continue
        for seed in SEEDS:
            torch.manual_seed(seed)
            x, rest = mk()
            if regime != "normal":
                x = x * scale
            try:
                ag = autograd_row_norms(fn, x, rest)
                cf = CLOSED[op](op, x, list(rest)).to(ag.dtype)
                rel = ((cf - ag).abs() / ag.abs().clamp_min(1e-12)).max().item()
                prof = ag[torch.isfinite(ag)]
                nz = prof[prof > 0]
                rows.append(dict(
                    op=op, regime=regime, seed=seed, m=int(ag.numel()),
                    max_rel_err=rel,
                    zero_frac=float((prof == 0).to(DT).mean().item()),
                    spread=float((nz.max() / nz.median()).item()) if nz.numel() else float("nan"),
                    L=float(prof.max().item()),
                ))
            except Exception as e:
                rows.append(dict(op=op, regime=regime, seed=seed,
                                 error=f"{type(e).__name__}: {e}"))

with open(os.path.join(HERE, "phase1_derivations.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

good = [r for r in rows if "error" not in r]
print(f"invocations: {len(rows)}   errors: {len(rows)-len(good)}")
worst = max(good, key=lambda r: r["max_rel_err"])
print(f"WORST max_rel_err across all: {worst['max_rel_err']:.3e} "
      f"({worst['op']}, {worst['regime']}, seed {worst['seed']})")
print(f"all within 1e-6: {all(r['max_rel_err'] < 1e-6 for r in good)}")
print()

print("=== Q2: profile degeneracy (saturating regime, mean over seeds) ===")
print(f"{'operator':20s} {'zero_frac':>10s} {'max/median':>11s}  note")
print("-" * 66)
ops = [o for o in REFS if any(r["op"] == o and r["regime"] == "saturating" for r in good)]
for op in ops:
    rs = [r for r in good if r["op"] == op and r["regime"] == "saturating"]
    zf = st.mean(r["zero_frac"] for r in rs)
    sp = [r["spread"] for r in rs if math.isfinite(r["spread"])]
    spm = st.mean(sp) if sp else float("nan")
    note = ""
    if zf >= 0.9:   note = "DEGENERATE -- profile is almost all zeros"
    elif zf >= 0.4: note = "sparse -- most rows dead"
    elif spm > 20:  note = "wide spread"
    print(f"{op:20s} {zf:10.3f} {spm:11.2f}  {note}")
