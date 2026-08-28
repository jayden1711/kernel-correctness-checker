"""
BLIND GENERALIZATION TEST, stage 3 (CPU): score measurement against the
pre-registered predictions. Reads both banked files; edits neither.

Reports, per config:
  z            = (y_meas - y_pred_mean) / y_pred_sd    (distributional law)
  paired ratio r_k = s_meas_k / (sigma-scaled s_lin_k) -- median and worst
                 |log10 r| (the Gram screen's own statistic, applied to a
                 kernel and operator the screen has never seen)
  m3/meas      -- whether the independence baseline over-predicts as the
                 correlated-row classification requires

Verdict criteria (stated before the GPU run; see METHOD.md):
  PASS if (i) all |z| <= 3 with family mean |z| consistent with noise,
          (ii) every median |log10 r| < log10(2) -- i.e. the shipped kernel
               is IN SCOPE at every tested input, including the two extreme
               regimes, per the pre-registered expectation,
          (iii) m3/meas > 1 wherever m3/gram > 1.05 (classification real).
Any other outcome is reported as-is.
"""

import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
THRESH = math.log10(2.0)


def q95(v):
    v = sorted(v)
    h = 0.95 * (len(v) - 1)
    lo = int(math.floor(h))
    fr = h - lo
    return v[lo] * (1 - fr) + v[min(lo + 1, len(v) - 1)] * fr


def main():
    bank = json.load(open(os.path.join(DATA, "blind_predictions.json")))
    delta_scale = bank["delta_scale"]
    preds = {p["key"]: p for p in bank["preds"]}
    meas = {}
    with open(os.path.join(DATA, "blind_gpu.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            meas[d["key"]] = d

    print(f"{'config':<26} {'y_meas':>8} {'y_pred':>8} {'z':>6} "
          f"{'med|lg r|':>9} {'max|lg r|':>9} {'m3/meas':>8}")
    zs, viol = [], []
    for key, p in sorted(preds.items()):
        m = meas.get(key)
        if m is None:
            print(f"{key:<26} MISSING MEASUREMENT")
            continue
        # The delta std is delta_scale * std(x) -- the y convention divides by
        # the DELTA sigma (the checker's own normalisation), not by std(x).
        sig = delta_scale * p["sigma"]
        y_meas = q95(m["s_meas"]) / (sig * p["L"])
        z = (y_meas - p["y_pred_mean"]) / p["y_pred_sd"]
        zs.append(z)
        logs = []
        for sm, sl in zip(m["s_meas"], p["s_lin"]):
            if sm > 0 and sl > 0:
                logs.append(math.log10(sm / sl))
        med = statistics.median(logs)
        worst = max(abs(v) for v in logs)
        if abs(med) >= THRESH:
            viol.append(key)
        m3_meas = p["y_m3_mean"] / y_meas
        print(f"{key:<26} {y_meas:8.4f} {p['y_pred_mean']:8.4f} {z:6.2f} "
              f"{abs(med):9.4f} {worst:9.4f} {m3_meas:8.4f}")

    n = len(zs)
    print(f"\nmean z = {statistics.mean(zs):+.3f} (expected sd "
          f"{1 / math.sqrt(n):.3f}), worst |z| = {max(abs(z) for z in zs):.2f} "
          f"over {n} configs")
    print(f"gram-screen violations (median |log10 r| >= log10 2): "
          f"{viol if viol else 'NONE -- kernel in scope at every tested input'}")
    m3g = [p["m3_over_gram"] for p in preds.values()]
    print(f"m3/gram structural factor: median {statistics.median(m3g):.3f}, "
          f"range [{min(m3g):.3f}, {max(m3g):.3f}]")


if __name__ == "__main__":
    main()
