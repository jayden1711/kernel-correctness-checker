"""H3, corrected pass — the two-arm resolvability criterion with the MEDIAN
s/ulp statistic (the one the scope-detect round validated), not the banked
min-based `s_over_ulp` (which the scope round showed overlaps for in-scope
operators; taxonomy_cv.py's first H3 pass used it by mistake and its m=1
flags are an artifact of that).

ulp recovery for phase1_native records: the probe computed
s_over_ulp = min(sens)/ulp, and sens is banked, so ulp = min(sens)/s_over_ulp
whenever min(sens) > 0. attn_native banks s_med and ulp directly.
"""

import json
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "../..")
K1 = 32.0
M1 = {"mse_loss", "huber_loss", "bce_loss", "kldiv_loss", "nll_loss",
      "cross_entropy"}
CAT1 = {"equal_attention_weights"}
CAT2 = {"skip_rescaling", "last_tile_dropped"}


def main():
    rows = []
    p = os.path.join(RUNS, "adaptive_tol_theory_2026-08-25/native_run/attn_native.jsonl")
    for r in map(json.loads, open(p)):
        rows.append(dict(op=r["op"], var=r["variant"], xmed=r["s_med"] / r["ulp"],
                         tol=r["tol"], src="attn"))
    unrec = 0
    p = os.path.join(RUNS, "phase1_derivations_2026-08-27/native_run/phase1_native.jsonl")
    for r in map(json.loads, open(p)):
        s = np.array(r["sens"])
        mn = s.min()
        if mn > 0 and r["s_over_ulp"] and r["s_over_ulp"] > 0:
            ulp = mn / r["s_over_ulp"]
            rows.append(dict(op=r["op"], var="primary", xmed=float(np.median(s)) / ulp,
                             tol=r["tol"], src="p1"))
        else:
            unrec += 1
    print(f"{len(rows)} invocations with median-based s/ulp ({unrec} unrecoverable)")

    groups = [("category-1 (abs floor)", lambda r: r["var"] in CAT1),
              ("category-2 (fp32 floor)", lambda r: r["var"] in CAT2),
              ("m=1 losses", lambda r: r["op"] in M1),
              ("everything else",
               lambda r: r["var"] not in CAT1 | CAT2 and r["op"] not in M1)]
    for grp, sel in groups:
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        flag = [r for r in g if min(r["xmed"] / K1, r["tol"] / 1e-6) <= 1.0]
        xs = [r["xmed"] for r in g]
        ts = [r["tol"] / 1e-6 for r in g]
        print(f"{grp:26s} n={len(g):3d} flagged={len(flag):3d}  "
              f"s_med/ulp range [{min(xs):.1f}, {max(xs):.1e}]  "
              f"tol/floor min {min(ts):.2f}")
        for r in flag:
            if r["op"] in M1 or (r["var"] not in CAT1 | CAT2 and r["op"] not in M1):
                print(f"    flagged: {r['op']}/{r['var']} s_med/ulp {r['xmed']:.1f} "
                      f"tol/floor {r['tol']/1e-6:.2f}")


if __name__ == "__main__":
    main()
