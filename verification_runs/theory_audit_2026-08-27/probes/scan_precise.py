"""H1 precision pass: pin the exact-law predictions tightly enough to judge the
-2.5% deviations on cumsum_exclusive and masked_cumsum, and measure the
discrete-walk correction constant precisely.

Adds per-invocation z-scores of every banked measurement against the predicted
(mu, sd), and non-power-of-two out-of-sample tests for all three unmasked ops.
"""

import json
import math
import os
import numpy as np

import scan_brownian as sb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "../data/scan_precise.json")

NREP = 1500

res = {}

meas = sb.load_measured()
adv = sb.load_adv()

print("=== high-precision exact-law predictions (NREP=%d) ===" % NREP)
for op in ["cumsum", "cumsum_reverse", "cumsum_exclusive", "masked_cumsum"]:
    mu, sd = sb.estimator_mean(op, nrep=NREP)
    sem_sim = sd / math.sqrt(NREP)
    ys = np.array([r["y"] for r in meas[op]])
    # predicted sd of the 6-invocation measured mean is sd/sqrt(6)
    z_mean = (ys.mean() - mu) / (sd / math.sqrt(len(ys)))
    z_each = (ys - mu) / sd
    print(f"{op:18s} pred {mu:.4f} +- {sd:.4f} (sim sem {sem_sim:.4f})  "
          f"meas mean {ys.mean():.4f}  meas/pred {ys.mean()/mu:.4f}  "
          f"z_mean {z_mean:+.2f}")
    print(f"{'':18s} per-invocation z: " + " ".join(f"{z:+.2f}" for z in z_each))
    res[op] = dict(mu=mu, sd=sd, meas=list(ys), z_mean=z_mean)

print("\n=== out-of-sample C=333 for all three unmasked ops ===")
for op in ["cumsum", "cumsum_reverse", "cumsum_exclusive"]:
    mu, sd = sb.estimator_mean(op, ncols=333, nrep=800)
    Lc = math.sqrt(333) if op != "cumsum_exclusive" else math.sqrt(332)
    rows = [r for r in adv if r["op"] == op and r.get("variant") == "non_power_of_two"]
    for r in rows:
        ym = r["tol"] / (3 * r["sigma"] * Lc)
        print(f"{op:18s} measured {ym:.4f}  pred {mu:.4f} +- {sd:.4f}  "
              f"z {(ym-mu)/sd:+.2f}")
        res[f"{op}_c333"] = dict(mu=mu, sd=sd, ym=ym)

print("\n=== discrete-walk correction ladder, high precision ===")
ycf = sb.closed_form_y()
print(f"continuous closed form y_inf = {ycf:.4f}")
lad = {}
for nc in [64, 128, 256, 512, 1024, 2048]:
    mu, sd = sb.estimator_mean("cumsum", ncols=nc, nrep=800)
    sem = sd / math.sqrt(800)
    gap = ycf - mu
    print(f"  C={nc:5d}  pred {mu:.4f} (sem {sem:.4f})  gap {gap:+.4f}  "
          f"gap*sqrt(C) {gap*math.sqrt(nc):+.3f} +- {sem*math.sqrt(nc):.3f}")
    lad[nc] = dict(mu=mu, sem=sem, gap_sqrtC=gap * math.sqrt(nc))
res["ladder"] = lad
res["y_continuous"] = ycf
res["siegmund_note"] = "beta = -zeta(1/2)/sqrt(2pi) = 0.5826, cf. measured gap*sqrt(C)"

json.dump(res, open(OUT, "w"), indent=1)
print("\nbanked ->", OUT)
