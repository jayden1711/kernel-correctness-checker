"""Re-run the autograd check against the SHIPPED verification/layer2_numeric_oracle/
structural_l.py, not the scratchpad prototype. Also smoke-tests that the 27
pre-existing operators still return a finite profile of the right length."""
import os, sys, math, json, statistics as st
import torch, torch.nn.functional as F

ROOT = "/Users/jaydenvasquez/Library/CloudStorage/GoogleDrive-jaydenvasquez1711@gmail.com/My Drive/kernel-correctness-checker"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification.layer2_numeric_oracle.structural_l import (
    row_norms, SUPPORTED_OPS, STATIC_OPS)
from derive import REFS, autograd_row_norms, DT

print(f"SUPPORTED_OPS: {len(SUPPORTED_OPS)}   STATIC_OPS: {len(STATIC_OPS)}")
print()

SEEDS = [0, 1, 2, 3, 4]
REGIMES = {"normal": 1.0, "wide": 8.0, "saturating": 40.0}
DOMAIN_LOCKED = {"bce_loss", "kldiv_loss", "nll_loss", "masked_cumsum"}

rows, fails = [], []
for op, (fn, mk) in REFS.items():
    ship_op = "rope" if op == "rope_nonunit" else op
    if ship_op not in SUPPORTED_OPS:
        print(f"  !! {ship_op} NOT in SUPPORTED_OPS"); continue
    for regime, scale in REGIMES.items():
        if op in DOMAIN_LOCKED and regime != "normal":
            continue
        for seed in SEEDS:
            torch.manual_seed(seed)
            x, rest = mk()
            if regime != "normal":
                x = x * scale
            ag = autograd_row_norms(fn, x, rest)
            cf = row_norms(ship_op, x, list(rest)).to(ag.dtype)
            rel = ((cf - ag).abs() / ag.abs().clamp_min(1e-12)).max().item()
            rows.append(dict(op=op, shipped_op=ship_op, regime=regime, seed=seed,
                             m=int(ag.numel()), max_rel_err=rel,
                             zero_frac=float((ag == 0).to(DT).mean().item()),
                             L=float(ag.max().item())))
            if rel >= 1e-6:
                fails.append((op, regime, seed, rel))

print(f"SHIPPED-MODULE CHECK: {len(rows)} invocations, {len(fails)} failures")
w = max(rows, key=lambda r: r["max_rel_err"])
print(f"worst max_rel_err: {w['max_rel_err']:.3e}  ({w['op']}/{w['regime']}/seed{w['seed']})")
for f in fails: print("  FAIL", f)

# --- regression smoke test on the pre-existing 27 -------------------------
print("\n=== regression: pre-existing operators still produce a profile ===")
dev = "cpu"
PREEX = {
 "softmax":      (torch.randn(8, 16, dtype=DT), []),
 "log_softmax":  (torch.randn(8, 16, dtype=DT), []),
 "layernorm":    (torch.randn(8, 16, dtype=DT), [torch.ones(16, dtype=DT)]),
 "rmsnorm":      (torch.randn(8, 16, dtype=DT), [torch.ones(16, dtype=DT)]),
 "matmul":       (torch.randn(8, 16, dtype=DT), [torch.randn(16, 12, dtype=DT)]),
 "gelu":         (torch.randn(8, 16, dtype=DT), []),
 "swish":        (torch.randn(8, 16, dtype=DT), []),
 "l1norm":       (torch.randn(8, 16, dtype=DT), []),
 "l2norm":       (torch.randn(8, 16, dtype=DT), []),
 "frobenius_norm":(torch.randn(8, 16, dtype=DT), []),
 "sum_reduction":(torch.randn(8, 16, dtype=DT), []),
 "mean_reduction":(torch.randn(8, 16, dtype=DT), []),
 "max_reduction":(torch.randn(8, 16, dtype=DT), []),
 "min_reduction":(torch.randn(8, 16, dtype=DT), []),
 "avg_pool1d":   (torch.randn(2, 3, 32, dtype=DT), [2, 2, 0]),
 "max_pool1d":   (torch.randn(2, 3, 32, dtype=DT), [2, 2, 0]),
 "cross_entropy":(torch.randn(8, 16, dtype=DT), [torch.randint(0, 16, (8,))]),
 "flash_attention":(torch.randn(8, 16, dtype=DT), [torch.randn(8,16,dtype=DT), torch.randn(8,16,dtype=DT)]),
}
bad = 0
for op, (x, rest) in PREEX.items():
    try:
        rn = row_norms(op, x, rest)
        ok = rn is not None and rn.numel() > 0 and torch.isfinite(rn).all()
        print(f"  {op:20s} m={rn.numel():5d}  L={rn.max().item():.4e}  {'ok' if ok else 'BAD'}")
        bad += 0 if ok else 1
    except Exception as e:
        print(f"  {op:20s} ERROR {type(e).__name__}: {e}"); bad += 1
print(f"regression failures: {bad}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "phase1_shipped.jsonl"), "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
