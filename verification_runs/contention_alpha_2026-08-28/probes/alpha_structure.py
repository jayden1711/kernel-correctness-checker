"""
Is the Fréchet alpha = 2.02 derivable, or coincidental?

The contention_tail round measured two estimates that disagree: Hill on the
top order statistics gives alpha = 2.9-3.7, while the two-point max growth
(23.26 @ n=560 -> 51.24 @ n=2765) gives alpha = 2.02, and the round leaned
on the latter for "the deepest tail behaves like alpha ~ 2". This probe asks
three sharp questions of the SAME banked data
(../../forkserver_2026-08-21/race_rate.jsonl):

  Q1 ARM MIXTURE. Both record maxima are spawn-arm. Are the two arms'
     tails actually different (per-arm Hill + bootstrap CI), so that the
     pooled deep tail is the spawn arm's and the pooled Hill is a mixture
     artifact?

  Q2 IS 2.02 MEASURABLE AT ALL? Under a null where the tail truly has
     alpha = alpha0 (semi-parametric: resample the body, Pareto tail above
     p99 with index alpha0), what is the sampling distribution of the
     two-point estimator alpha_hat = ln(n2/n1)/ln(M2/M1)? If its spread
     covers 2.02 from alpha0 = 3+, the "alpha ~ 2 deep tail" claim
     dissolves into max-draw noise and must be retracted to a range.

  Q3 SHAPE DISCRIMINATION. Pareto vs lognormal tail on each arm
     (log-log tail linearity; likelihood ratio on exceedances above p95):
     can this dataset even distinguish the families? A negative here
     bounds what ANY alpha derivation could claim from this data.

The DERIVATION question (is there a scheduling-structure argument forcing a
specific alpha?) is addressed in the FINDINGS with the candidate mechanisms
and what each predicts; this probe supplies the measurements those
candidates must survive.

Run:  .venv/bin/python alpha_structure.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(ROOT, "verification_runs", "forkserver_2026-08-21",
                   "race_rate.jsonl")
DATA = os.path.join(HERE, "..", "data")


def hill(x, k):
    """Hill estimator on the top-k order statistics."""
    s = np.sort(x)[::-1]
    if k >= len(s):
        return None
    top = s[:k]
    xk = s[k]
    return 1.0 / np.mean(np.log(top / xk))


def two_point(x, n1):
    """alpha_hat from max growth between prefix n1 and the full sample,
    in RECORD ORDER (the estimator the round actually used)."""
    m1 = np.max(x[:n1])
    m2 = np.max(x)
    if m2 <= m1:
        return None
    return math.log(len(x) / n1) / math.log(m2 / m1)


def main():
    os.makedirs(DATA, exist_ok=True)
    rows = [json.loads(l) for l in open(SRC)]
    ratios = {"spawn": [], "forkserver": []}
    order = []
    for r in rows:
        if r.get("ratio") is not None and r.get("arm") in ratios:
            ratios[r["arm"]].append(r["ratio"])
            order.append(r["ratio"])
    order = np.array(order)
    pooled = order
    rng = np.random.default_rng(7)
    print(f"pooled n = {len(pooled)}  "
          f"spawn n = {len(ratios['spawn'])}  "
          f"forkserver n = {len(ratios['forkserver'])}")
    print(f"pooled max = {pooled.max():.2f}, "
          f"second/third = {np.sort(pooled)[-2]:.2f}/"
          f"{np.sort(pooled)[-3]:.2f}")
    for arm in ("spawn", "forkserver"):
        a = np.array(ratios[arm])
        print(f"{arm}: max = {a.max():.2f}  p99 = "
              f"{np.percentile(a, 99):.2f}  p95 = "
              f"{np.percentile(a, 95):.2f}")

    # ---------------- Q1: per-arm Hill with bootstrap CIs -----------------
    print("\n== Q1: per-arm Hill (k = 25, 50, 100, 200) ==")
    hill_tab = {}
    for name, a in [("pooled", pooled)] + list(
            (k, np.array(v)) for k, v in ratios.items()):
        a = np.asarray(a)
        row = {}
        for k in (25, 50, 100, 200):
            h = hill(a, k) if k < len(a) else None
            if h is None:
                row[k] = None
                continue
            boots = []
            for _ in range(2000):
                b = rng.choice(a, size=len(a), replace=True)
                hb = hill(b, k)
                if hb:
                    boots.append(hb)
            lo, hi_ = np.percentile(boots, [2.5, 97.5])
            row[k] = (h, lo, hi_)
            print(f"  {name:10s} k={k:3d}: alpha = {h:.2f}  "
                  f"[{lo:.2f}, {hi_:.2f}]")
        hill_tab[name] = {k: (list(v) if v else None)
                          for k, v in row.items()}

    # ---------------- Q2: two-point estimator sampling law ----------------
    print("\n== Q2: two-point alpha_hat under alpha0 nulls ==")
    n1, n2 = 560, len(pooled)
    obs = two_point(pooled, n1)
    print(f"  observed two-point alpha_hat (n1=560) = {obs:.2f}")
    body = pooled[pooled <= np.percentile(pooled, 99)]
    p99 = np.percentile(pooled, 99)
    for alpha0 in (2.0, 2.5, 3.0, 3.5):
        hats = []
        for _ in range(4000):
            n_tail = rng.binomial(n2, 0.01)
            tail = p99 * (1 - rng.random(n_tail)) ** (-1.0 / alpha0)
            samp = np.concatenate([
                rng.choice(body, size=n2 - n_tail, replace=True), tail])
            rng.shuffle(samp)
            h = two_point(samp, n1)
            if h is not None:
                hats.append(h)
        hats = np.array(hats)
        p_le = float(np.mean(hats <= obs))
        print(f"  alpha0 = {alpha0}: alpha_hat p5/p50/p95 = "
              f"{np.percentile(hats,5):.2f}/{np.percentile(hats,50):.2f}/"
              f"{np.percentile(hats,95):.2f}   P(alpha_hat <= {obs:.2f}) "
              f"= {p_le:.3f}")

    # ---------------- Q3: Pareto vs lognormal above p95 -------------------
    print("\n== Q3: exceedance family discrimination (per arm, u = p95) ==")
    for arm in ("spawn", "forkserver"):
        a = np.array(ratios[arm])
        u = np.percentile(a, 95)
        exc = a[a > u]
        n = len(exc)
        # Pareto MLE on exceedances (relative): alpha = 1/mean(log(x/u))
        alpha_mle = 1.0 / np.mean(np.log(exc / u))
        llp = n * math.log(alpha_mle) + n * alpha_mle * math.log(u) \
            - (alpha_mle + 1) * np.sum(np.log(exc))
        # lognormal MLE on log(exc) truncated at log u: fit normal to
        # log(exc) by truncated-normal MLE (coarse grid)
        lx = np.log(exc)
        best = None
        for mu in np.linspace(lx.mean() - 2, lx.mean() + 2, 81):
            for s in np.linspace(0.05, 3.0, 60):
                z = (lx - mu) / s
                zu = (math.log(u) - mu) / s
                # truncated normal density on log-scale, Jacobian 1/x
                from math import erf, sqrt
                tail_mass = 0.5 * (1 - erf(zu / sqrt(2)))
                if tail_mass <= 1e-12:
                    continue
                ll = (-0.5 * np.sum(z**2) - n * math.log(s)
                      - n * 0.5 * math.log(2 * math.pi)
                      - np.sum(lx)                # Jacobian
                      - n * math.log(tail_mass))
                if best is None or ll > best[0]:
                    best = (ll, mu, s)
        lll = best[0]
        print(f"  {arm}: n_exc = {n}, pareto alpha_MLE = {alpha_mle:.2f}, "
              f"logL pareto = {llp:.1f}, logL lognorm = {lll:.1f}, "
              f"delta = {llp - lll:+.1f} "
              f"({'pareto' if llp > lll else 'lognormal'} preferred)")

    json.dump({"hill": hill_tab, "two_point_obs": obs},
              open(os.path.join(DATA, "alpha_structure.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
