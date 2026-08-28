"""
Two follow-ups on the A/B arms:

1. TRUE DIRECT COVERAGE. The arm analysis undercounted (the estimator is
   named only in FAIL messages). Recover it from the banked records by
   comparing per-record adaptive_tol between arms: same seeds, so a record
   whose tol differs between A and D took the direct path (E vs draw); a
   record with identical tol either fell back (identical probe draw) or is
   floor-clamped in both (ambiguous, counted separately).

2. E-vs-DRAW VALIDATION AT CORPUS SCALE. The ratio tol_D/tol_A on
   direct-taken records is the parent mean over the measured q95 draw —
   its distribution should sit near 1 with the q95-of-20 sampling spread
   (~7-14% CV), replicating the native-bank validation on the full
   GPU corpus call mix including adversarial variants.

Run:  .venv/bin/python tol_ratio_coverage.py
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.path.join(HERE, "..", "data", "gpu")
RE_TOL = re.compile(r"adaptive_tol=(\d+\.\d+)")

ATT = {"flash_attention", "causal_flash_attention",
       "scaled_dot_product_attention"}


def tols(d):
    out = {}
    for e in d["entries"]:
        for tag, rec in [("mutant", e["mutant"])] + [
                (f"ref{i}", r) for i, r in enumerate(e["refs"])]:
            for x in rec["records"]:
                nm = x["name"]
                if nm == "perturbation_tolerance" or nm.startswith(
                        "adversarial"):
                    m = RE_TOL.search(str(x.get("detail", "")))
                    if m:
                        out[(e["op"], e["mutant"]["name"], tag, nm)] = \
                            float(m.group(1))
    return out


def main():
    A = tols(json.load(open(os.path.join(GPU, "A_ver.json"))))
    D = tols(json.load(open(os.path.join(GPU, "D_ver.json"))))
    keys = sorted(set(A) & set(D))
    print(f"paired perturbation-routed records with printed tol: {len(keys)}")
    differ, same_floor, same_other = [], [], []
    excluded = []
    for k in keys:
        a, d = A[k], D[k]
        if k[0] in ATT or k[0] in ("argmax", "argmin"):
            excluded.append(k)
        elif a != d:
            differ.append(k)
        elif a == 1e-6:
            same_floor.append(k)
        else:
            same_other.append(k)
    print(f"excluded ops (attention/arg*): {len(excluded)}")
    print(f"tol differs (direct TAKEN, unambiguous): {len(differ)}")
    print(f"tol identical at floor (ambiguous -- either path clamps): "
          f"{len(same_floor)}")
    print(f"tol identical off floor (probe fallback or coincidence): "
          f"{len(same_other)}")
    per_op = defaultdict(lambda: [0, 0])
    for k in keys:
        if k[0] in ATT or k[0] in ("argmax", "argmin"):
            continue
        per_op[k[0]][1] += 1
        if A[k] != D[k] or A[k] == 1e-6:
            per_op[k[0]][0] += 1
    fallback_like = {op: f"{c[1]-c[0]}/{c[1]}" for op, c in per_op.items()
                     if c[0] < c[1]}
    print("non-excluded ops with off-floor identical tol:", fallback_like)

    ratios = np.array([D[k] / A[k] for k in differ])
    lr = np.log(ratios)
    print(f"\nE/draw ratio on direct-taken records (n={len(ratios)}): "
          f"p5/p50/p95 = {np.percentile(ratios,5):.3f}/"
          f"{np.percentile(ratios,50):.3f}/{np.percentile(ratios,95):.3f}  "
          f"log-sd {lr.std():.3f} dex-equivalent CV ~{lr.std():.1%}")
    worst = sorted(differ, key=lambda k: abs(np.log(D[k]/A[k])))[-5:]
    for k in worst:
        print(f"  extreme: {k}  A={A[k]:.6g}  D={D[k]:.6g}  "
              f"ratio {D[k]/A[k]:.3f}")


if __name__ == "__main__":
    main()
