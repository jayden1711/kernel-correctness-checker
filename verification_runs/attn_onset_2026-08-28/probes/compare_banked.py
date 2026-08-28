"""
Banked-GPU-arm side of the onset-law validation: pools the per-delta Gram
ratios of the large_magnitude_qk records from the CURRENT banked arm
(layernorm_mask_fix round, G_lnfix -- attention records identical in every
arm since the attention mask fix) and prints their quantiles next to the
analytic pushforward at tau = sqrt(2)*1e-3*20^2, plus the per-record
medians and floor statistics.

Run:  .venv/bin/python compare_banked.py
"""
import gzip
import json
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.path.join(HERE, "..", "..", "layernorm_mask_fix_2026-08-28",
                   "arms", "G_lnfix.json.gz")
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
TAU = math.sqrt(2) * 1e-3 * 400


def phi(u):
    return 1.0 if u == 0 else (1.0 - math.exp(-u)) / u


def quant(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    d = json.load(gzip.open(ARM, "rt"))
    pooled, meds, sulps = {}, {}, {}
    for e in d["entries"]:
        op = e["op"]
        if op not in ("causal_flash_attention", "scaled_dot_product_attention"):
            continue
        packs = [e["mutant"]["records"]] + [r["records"] for r in e["refs"]]
        for recs in packs:
            for r in recs:
                if r["name"] != "adversarial_large_magnitude_qk":
                    continue
                for sc in (r.get("scope_flags") or []):
                    if sc.get("kind") != "scope_divergence":
                        continue
                    rat = sc.get("gram_log10_ratios") or []
                    pooled.setdefault(op, []).extend(10 ** x for x in rat)
                    if sc.get("gram_log10_median") is not None:
                        meds.setdefault(op, []).append(
                            10 ** sc["gram_log10_median"])
                    sulps.setdefault(op, []).append(sc.get("sulp_median"))

    g = torch.Generator().manual_seed(7)
    mc = [phi(float(torch.randn(1, generator=g)) * TAU) for _ in range(200000)]

    print(f"analytic pushforward (tau={TAU:.4f}, parameter-free):")
    print("  " + "  ".join(f"P{int(p*100):02d}={quant(mc, p):.3f}" for p in QS))
    for op, rr in pooled.items():
        print(f"\n{op}: {len(rr)} banked per-delta ratios "
              f"({len(meds.get(op, []))} records with medians)")
        print("  " + "  ".join(f"P{int(p*100):02d}={quant(rr, p):.3f}" for p in QS))
        print(f"  medians: {[round(m, 3) for m in sorted(meds.get(op, []))]}")
        print(f"  s/ulp:   {[round(s, 1) for s in sorted(sulps[op])]}")
    n_floor = sum(1 for ss in sulps.values() for s in ss if s is not None and s < 32)
    n_tot = sum(len(ss) for ss in sulps.values())
    print(f"\nfloor-flagged: {n_floor}/{n_tot} banked large_magnitude_qk records")


if __name__ == "__main__":
    main()
