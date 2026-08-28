"""
KINK BOUND for the Gram screen -- turning the measured 1.44x into g(p).

Setting: l1norm f(x)_rj = x_rj / (sum_k |x_rk| + eps), evaluated at an input
whose rows have a fraction p of entries EXACTLY zero (the corpus's
second_half_dominant adversarial variant: p = 1/2, C = 128, nonzeros
~ N(0, 10^2)). |.| is not differentiable at 0, and the autograd Jacobian
(sign(0) = 0) misses the response of the zero coordinates in the
DENOMINATOR.

First-order decomposition (derived in FINDINGS.md, validated in part 3
below): with D = sum|x| + eps, A = sum_{nonzero} sign(x_k) d_k,
K = sum_{zero} |d_k| >= 0,

    f(x+d) - f(x)  =  d/D - x (A + K)/D^2 + O(sigma^2)
    J d            =  d/D - x A/D^2                      (sign(0) = 0)

so the screen's per-delta ratio deviates through the RECTIFIED sum K alone:
E K = pC sigma sqrt(2/pi), first order in sigma. Both sides are Theta(sigma),
hence the ratio is SCALE-INVARIANT -- the deviation is geometry, not noise,
and no delta_scale makes it go away. That is why the corpus measures a fixed
1.44x.

This probe:
  1. replicates the shipped screen statistic (median log10 s_meas/s_lin over
     20 deltas, torch.func.jvp, float64) against the float64 math function
     itself -- no kernel, no fp32 -- at the exact corpus configuration, and
     compares with the five banked G-arm medians;
  2. validates the first-order decomposition delta-by-delta;
  3. sweeps p to produce g(p) = median screen statistic as a function of the
     kink fraction, locates p* where g crosses the factor-2 flag line, and
     checks the asymptotic scaling (g - 1 proportional to p/(1-p) at fixed C);
  4. verifies scale-invariance in delta_scale and nonzero magnitude tau.

Writes data/kink_bound.json.
"""

import json
import math
import os
import statistics
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
DATA = os.path.join(HERE, "..", "data")

EPS = 1e-12
R, C = 64, 128
TAU = 10.0
NDELTA = 20
BANKED = [0.12764, 0.132858, 0.145486, 0.154145, 0.158634]  # G-arm medians


def f(x):
    return x / (x.abs().sum(dim=-1, keepdim=True) + EPS)


def make_x(rng, p, tau=TAU, rows=R, cols=C):
    x = torch.zeros(rows, cols, dtype=torch.float64)
    nz = cols - int(round(p * cols))
    vals = torch.from_numpy(rng.standard_normal((rows, nz)) * tau)
    x[:, :nz] = vals            # contiguous zero block, like the variant
    return x


def screen_stat(x, rng, delta_scale=1e-3, ndelta=NDELTA):
    """Median log10 (s_meas/s_lin) -- the shipped Gram-screen statistic,
    with the float64 math function standing in for the kernel."""
    sigma = delta_scale * float(x.to(torch.float32).std())
    logs = []
    for _ in range(ndelta):
        d = torch.from_numpy(rng.standard_normal(tuple(x.shape))) * sigma
        s_meas = float((f(x + d) - f(x)).abs().max())
        _, jd = torch.func.jvp(f, (x,), (d,))
        s_lin = float(jd.abs().max())
        if s_meas > 0 and s_lin > 0:
            logs.append(math.log10(s_meas / s_lin))
    return statistics.median(logs)


def first_order_check(x, rng, delta_scale=1e-3, ndelta=8):
    """|exact - first-order| / exact, per delta."""
    sigma = delta_scale * float(x.to(torch.float32).std())
    D = x.abs().sum(dim=-1, keepdim=True) + EPS
    sgn = torch.sign(x)
    errs = []
    for _ in range(ndelta):
        d = torch.from_numpy(rng.standard_normal(tuple(x.shape))) * sigma
        exact = f(x + d) - f(x)
        A = (sgn * d).sum(dim=-1, keepdim=True)      # linear part of dD
        K = (d * (x == 0)).abs().sum(dim=-1, keepdim=True)  # rectified part
        approx = d / D - x * (A + K) / (D * D)
        errs.append(float((exact - approx).abs().max() / exact.abs().max()))
    return max(errs)


def main():
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(2026)

    # 1 -- corpus configuration, 200 independent invocations
    stats = [screen_stat(make_x(rng, 0.5), rng) for _ in range(200)]
    stats.sort()
    med = statistics.median(stats)
    lo, hi = stats[4], stats[-5]   # ~2.5/97.5 percentiles
    inside = sum(1 for b in BANKED if stats[0] <= b <= stats[-1])
    print(f"corpus config (p=1/2, C=128, tau=10): median log10 r = {med:.4f} "
          f"(ratio {10**med:.3f}), 95% band [{lo:.4f}, {hi:.4f}] "
          f"(ratios [{10**lo:.3f}, {10**hi:.3f}])")
    print(f"banked G-arm medians {BANKED} -> {inside}/5 inside the simulated "
          f"range [{stats[0]:.4f}, {stats[-1]:.4f}]")

    # 2 -- first-order decomposition
    worst = max(first_order_check(make_x(rng, 0.5), rng) for _ in range(10))
    print(f"first-order decomposition: worst relative error {worst:.2e} "
          f"(over 10 invocations x 8 deltas)")

    # 3 -- g(p) sweep and p*
    P_GRID = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375]
    sweep = {}
    for p in P_GRID:
        vals = [screen_stat(make_x(rng, p), rng) for _ in range(60)]
        vals.sort()
        sweep[p] = dict(median=statistics.median(vals),
                        lo=vals[1], hi=vals[-2])
        m = sweep[p]["median"]
        print(f"  p={p:6.4f}  median log10 r = {m:+.4f}  ratio {10**m:6.3f}  "
              f"band [{10**vals[1]:.3f}, {10**vals[-2]:.3f}]  "
              f"(g-1)/(p/(1-p)) = "
              f"{(10**m - 1) / (p / (1 - p)) if p > 0 else float('nan'):.3f}")
    # p'*: first crossing of the median above log10 2, linear interpolation
    thr = math.log10(2.0)
    pstar = None
    ps = sorted(sweep)
    for a, b in zip(ps, ps[1:]):
        ma, mb = sweep[a]["median"], sweep[b]["median"]
        if ma < thr <= mb:
            pstar = a + (thr - ma) / (mb - ma) * (b - a)
            break
    print(f"p* (median crosses the factor-2 flag line): "
          f"{pstar:.3f}" if pstar else "p*: not crossed on grid")
    # conservative p*: where the UPPER band crosses (first invocation could fire)
    pfire = None
    for a, b in zip(ps, ps[1:]):
        ha, hb = sweep[a]["hi"], sweep[b]["hi"]
        if ha < thr <= hb:
            pfire = a + (thr - ha) / (hb - ha) * (b - a)
            break
    print(f"p_fire (upper band crosses -- earliest plausible fire): "
          f"{pfire:.3f}" if pfire else "p_fire: not crossed on grid")

    # 4 -- invariances
    x = make_x(np.random.default_rng(7), 0.5)
    inv_scale = [screen_stat(x, np.random.default_rng(11), ds)
                 for ds in (1e-4, 1e-3, 1e-2)]
    inv_tau = [screen_stat(make_x(np.random.default_rng(7), 0.5, tau=t),
                           np.random.default_rng(11)) for t in (0.1, 10.0, 1e3)]
    print(f"scale invariance (delta_scale 1e-4/1e-3/1e-2): "
          f"{[f'{10**v:.4f}' for v in inv_scale]}")
    print(f"tau invariance (0.1/10/1000): {[f'{10**v:.4f}' for v in inv_tau]}")

    json.dump(dict(corpus_median=med, corpus_band=[lo, hi],
                   corpus_all=stats, banked=BANKED,
                   first_order_worst=worst, sweep=sweep,
                   pstar=pstar, pfire=pfire,
                   inv_scale=inv_scale, inv_tau=inv_tau),
              open(os.path.join(DATA, "kink_bound.json"), "w"))
    print("wrote data/kink_bound.json")


if __name__ == "__main__":
    main()
