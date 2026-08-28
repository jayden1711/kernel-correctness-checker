"""Join the pre-registered CPU Gram-law predictions with the GPU out-of-sample
measurements and score the falsification tests:

  F1 out-of-sample shapes: y_meas vs y_pred at (128,64), (100,32), (256,16),
     none of which any banked round ever measured. Per-point z, per-shape and
     per-op aggregates.
  F2 scale invariance: within one (op, shape, seed) the three delta_scale arms
     reuse the same generator seed, so under the first-order law y must be
     IDENTICAL across arms up to fp noise -- the attention analogue of the
     scan family's four-decimal input-invariance test.
  F3 the corpus-shape anchor (64,32): continuity with the banked round.
"""

import json
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "../data")

preds = {(p["op"], p["N"], p["D"], p["seed"]): p
         for p in json.load(open(os.path.join(D, "attn_gram_predictions.json")))}
meas = [json.loads(l) for l in open(os.path.join(D, "attn_gram_gpu.jsonl"))]

rows = []
for r in meas:
    p = preds[(r["op"], r["N"], r["D"], r["seed"])]
    y = r["tol"] / (3 * r["sigma"] * p["L_exact"])
    rows.append(dict(op=r["op"], N=r["N"], Dh=r["D"], seed=r["seed"],
                     ds=r["delta_scale"], y=y, y_pred=p["y_pred"],
                     y_sd=p["y_sd"], z=(y - p["y_pred"]) / p["y_sd"],
                     m3=p["m3_pred"], defect=r["defect_t01"]))

print("=== F2 scale invariance (same deltas, three delta_scales) ===")
worst = 0.0
for key in sorted({(r["op"], r["N"], r["Dh"], r["seed"]) for r in rows}):
    ys = [r["y"] for r in rows if (r["op"], r["N"], r["Dh"], r["seed"]) == key]
    spread = max(ys) / min(ys) - 1
    worst = max(worst, spread)
print(f"36 triples: worst y spread across delta_scales = {worst:.3%}")

print("\n=== F1/F3 per-point (delta_scale=1e-3 arm) ===")
one = [r for r in rows if r["ds"] == 1e-3]
for r in one:
    print(f"{r['op']:30s} {r['N']:4d}x{r['Dh']:<3d} s{r['seed']} "
          f"y {r['y']:.4f} pred {r['y_pred']:.4f}+-{r['y_sd']:.4f} "
          f"z {r['z']:+.2f}")

print("\n=== aggregates over ALL 108 measurement points ===")
allz = np.array([r["z"] for r in rows])
ratio = np.array([r["y"] / r["y_pred"] for r in rows])
print(f"n={len(rows)}  meas/pred median {np.median(ratio):.4f} "
      f"[{ratio.min():.4f},{ratio.max():.4f}]  mean z {allz.mean():+.3f} "
      f"(expected sd of mean ~{1/math.sqrt(36):.2f}; 36 independent inputs) "
      f" worst |z| {abs(allz).max():.2f}")
for op in sorted({r["op"] for r in rows}):
    rs = [r for r in rows if r["op"] == op]
    rr = np.array([r["y"] / r["y_pred"] for r in rs])
    zz = np.array([r["z"] for r in rs])
    print(f"  {op:30s} n={len(rs):3d} median {np.median(rr):.4f} "
          f"mean z {zz.mean():+.2f} worst |z| {abs(zz).max():.2f}")
for (N, Dh) in sorted({(r["N"], r["Dh"]) for r in rows}):
    rs = [r for r in rows if (r["N"], r["Dh"]) == (N, Dh)]
    rr = np.array([r["y"] / r["y_pred"] for r in rs])
    zz = np.array([r["z"] for r in rs])
    tag = "corpus anchor" if (N, Dh) == (64, 32) else "OUT-OF-SAMPLE"
    print(f"  {N:4d}x{Dh:<3d} {tag:14s} n={len(rs):3d} median {np.median(rr):.4f} "
          f"mean z {zz.mean():+.2f} worst |z| {abs(zz).max():.2f}")

m3g = np.array([r["m3"] / r["y_pred"] for r in rows if r["ds"] == 1e-3])
print(f"\nm3/gram across 36 inputs: median {np.median(m3g):.4f} "
      f"[{m3g.min():.4f},{m3g.max():.4f}]")
json.dump(rows, open(os.path.join(D, "attn_gram_compare.json"), "w"), indent=1)
