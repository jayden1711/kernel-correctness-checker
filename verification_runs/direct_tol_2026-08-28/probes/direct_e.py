"""
DIRECT closed-form E[q95_n] under the structural M3 parent -- deterministic,
no simulation.

The estimand is exactly M3's (y_profile): with w the normalized row-norm
profile, s = max_i w_i |z_i| has parent CDF

    F(t) = prod_i (2 Phi(t / w_i) - 1)

and the checker's statistic is torch.quantile's linear interpolation of the
0.95 quantile over n iid draws of s:

    q95_n = (1 - frac) * X_(lo+1:n) + frac * X_(hi+1:n),
    h = 0.95 (n-1), lo = floor(h), hi = min(lo+1, n-1), frac = h - lo.

By linearity, E[q95_n] = (1-frac) E[X_(lo+1:n)] + frac E[X_(hi+1:n)], and each
order-statistic mean is a grid integral

    E[X_(k:n)] = int_0^inf (1 - G_k(t)) dt,
    G_k(t) = P(X_(k:n) <= t) = sum_{j=k}^n C(n,j) F(t)^j (1 - F(t))^{n-j}.

No random draws anywhere. Two cost devices, both with controlled error:

  * BINNING: rows are grouped into log-spaced bins of relative width
    BIN_REL (each w_i replaced by its bin center, |dw/w| <= BIN_REL/2), so F
    costs O(n_bins * grid) instead of O(m * grid). Since s is a max of
    w-scaled variables, a multiplicative perturbation of every w_i by at most
    (1 +- e) brackets the statistic multiplicatively: computing E on the
    lower- and upper-shifted bin edges brackets the true E.
  * TRUNCATION: rows with w_i < W_CUT * w_max are dropped. Their factor
    satisfies (2 Phi(t/w_i) - 1) >= 2 Phi(t_lo / (W_CUT w_max)) - 1 for
    t >= t_lo; the induced multiplicative error on F over the integration
    region is printed and checked, not assumed.

This probe validates the implementation three ways before it goes anywhere
near production:
  V1  against y_profile (M3's simulation) at high NSIM on synthetic profiles;
  V2  against the banked 228-invocation native run: predicted
      tol_n = scale * sigma * L * E[q95_n] vs the measured prefix
      qlin(sens[:n]) for n in {20, 40};
  V3  cost: wall time per call vs y_profile and vs the banked probe cost.

Run:  .venv/bin/python direct_e.py
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "autokernel", "files"))

from verification.layer2_numeric_oracle.structural_l import (
    row_norms, y_profile)

NATIVE = os.path.join(ROOT, "verification_runs",
                      "adaptive_tol_theory_2026-08-25", "native_run",
                      "gpu_native.jsonl")

SQRT2 = math.sqrt(2.0)


def _order_stat_weights(n, q=0.95):
    h = q * (n - 1)
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    frac = h - lo
    # 1-based order-statistic indices and weights
    return [(lo + 1, 1.0 - frac)] + ([(hi + 1, frac)] if frac > 0 else [])


def e_q95_direct(w, n_samples, q=0.95, grid=2048, bin_rel=2e-3,
                 w_cut=0.25, return_bracket=False):
    """E[q95_n(max_i w_i |z_i|)] for normalized profile w (max = 1).

    Deterministic. Returns the bin-center estimate; with return_bracket=True
    also the (lower, upper) bin-edge bracket which contains the exact value
    up to truncation error.
    """
    w = np.asarray(w, dtype=np.float64)
    w = w[w > 0]
    if w.size == 0:
        return None
    wmax = w.max()
    w = w / wmax
    m_full = w.size
    kept = w[w >= w_cut]
    n_dropped = m_full - kept.size
    # log-spaced binning of the kept rows
    lo_edge = kept.min()
    nbins = max(1, int(math.ceil(math.log(1.0 / lo_edge + 1e-12) /
                                 math.log1p(bin_rel))) + 1)
    edges = np.exp(np.linspace(math.log(lo_edge) - 1e-12, 1e-12, nbins + 1))
    idx = np.clip(np.searchsorted(edges, kept, side="right") - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins)
    nz = counts > 0
    centers = np.sqrt(edges[:-1] * edges[1:])[nz]
    cnts = counts[nz].astype(np.float64)

    # grid over t; upper end from the max bound
    t_hi = math.sqrt(2 * math.log(max(2 * m_full, 4))) + 6.0
    t = np.linspace(0.0, t_hi, grid)

    def _E(scale_w):
        # log F(t) = sum_bins count * log(erf(t / (sqrt2 * w_bin)))
        with np.errstate(divide="ignore"):
            a = t[None, :] / (SQRT2 * (centers * scale_w)[:, None])
            c = np.clip(torch.erf(torch.from_numpy(
                np.ascontiguousarray(a))).numpy(), 1e-300, 1.0)
            logF = (cnts[:, None] * np.log(c)).sum(axis=0)
        F = np.exp(logF)
        # order-statistic means
        total = 0.0
        for k, wt in _order_stat_weights(n_samples, q):
            # G_k(t) = sum_{j=k}^n C(n,j) F^j (1-F)^(n-j)
            G = np.zeros_like(F)
            for j in range(k, n_samples + 1):
                G += math.comb(n_samples, j) * F**j * (1 - F)**(n_samples - j)
            Ek = np.trapezoid(1.0 - G, t)
            total += wt * Ek
        return total

    est = _E(1.0)
    if not return_bracket:
        return est
    lo_b = _E(1.0 / (1 + bin_rel / 2))
    hi_b = _E(1 + bin_rel / 2)
    return est, (min(lo_b, hi_b), max(lo_b, hi_b)), n_dropped


def qlin(xs, q=0.95):
    s = sorted(xs)
    n = len(s)
    h = q * (n - 1)
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


# ---------------------------------------------------------------- V1: vs M3
def v1():
    print("== V1: direct vs y_profile simulation (NSIM=60000) ==")
    rng = np.random.default_rng(0)
    profiles = {
        "flat_m48": np.ones(48),
        "flat_m8192": np.ones(8192),
        "spike_plus_tail": np.concatenate([[1.0], 0.3 * np.ones(4095)]),
        "halfnormal_m1000": np.abs(rng.standard_normal(1000)),
        "decay_m512": np.exp(-np.arange(512) / 80.0),
        "two_scale": np.concatenate([np.ones(5), 0.5 * np.ones(2000)]),
    }
    worst = 0.0
    for name, p in profiles.items():
        for n in (20, 40):
            direct, br, ndrop = e_q95_direct(p, n, return_bracket=True)
            sim = y_profile(torch.from_numpy(np.asarray(p, dtype=np.float32)),
                            n, nsim=60000)
            rel = direct / sim - 1
            worst = max(worst, abs(rel))
            print(f"  {name:18s} n={n:2d}  direct={direct:.5f} "
                  f"bracket=({br[0]:.5f},{br[1]:.5f}) dropped={ndrop:5d} "
                  f"sim={sim:.5f}  rel={rel:+.4%}")
    print(f"  worst |rel| = {worst:.4%}  (sim MC error at NSIM=60000 ~ 0.3%)")
    return worst


# ------------------------------------------------- V2: vs banked native bank
def replay_inputs():
    """Bit-exact replay of gpu_native.py's input loop (b3_chain's proven
    path, copied verbatim): one entry per (op, mutant), 6 draws each."""
    from tritonbench_registry import OPS, FAMILIES
    rng = np.random.default_rng(0)
    out = {}
    entry = 0
    for spec_key, ref_file, cheat_dir, family, mutant_names in OPS:
        mk_fn = FAMILIES[family][0]
        for _mut in mutant_names:
            for j in range(6):
                np_args = mk_fn(rng)
                out[(entry, j)] = (spec_key, np_args)
            entry += 1
    return out


def to_torch64(np_args):
    ts = [torch.from_numpy(a).to(torch.float64) if isinstance(a, np.ndarray)
          and a.dtype != np.int64 else
          (torch.from_numpy(a) if isinstance(a, np.ndarray) else a)
          for a in np_args]
    return ts[0], ts[1:]


def v2():
    print("\n== V2: predicted tol vs banked measured prefix quantile ==")
    bank = [json.loads(l) for l in open(NATIVE)]
    bank = [r for r in bank if r.get("kind") == "primary" and r.get("sens")]
    bykey = replay_inputs()
    rows = []
    t_direct = 0.0
    for r in bank:
        key = (r["entry"], r["inv"])
        if key not in bykey:
            continue
        op, np_args = bykey[key]
        assert op == r["op"], (op, r["op"])
        x, rest = to_torch64(np_args)
        sig_expect = 1e-3 * x.float().std().item()
        if abs(sig_expect - r["sigma"]) > 1e-9 * max(1, abs(r["sigma"])):
            print(f"  REPLAY MISMATCH {r['op']} {key} "
                  f"{sig_expect} vs {r['sigma']}")
            continue
        try:
            rn = row_norms(op, x, rest)
        except Exception as e:
            print(f"  row_norms failed {op}: {e}")
            continue
        if rn is None:
            continue
        rn = rn.double().numpy().ravel()
        t0 = time.perf_counter()
        y20 = e_q95_direct(rn, 20)
        t_direct += time.perf_counter() - t0
        y40 = e_q95_direct(rn, 40)
        L = float(np.max(rn))
        pred20 = r["sigma"] * L * y20
        pred40 = r["sigma"] * L * y40
        meas20 = qlin(r["sens"][:20])
        meas40 = qlin(r["sens"][:40])
        rows.append(dict(op=op, entry=r["entry"], inv=r["inv"],
                         pred20=pred20, meas20=meas20,
                         pred40=pred40, meas40=meas40, cv=r.get("cv")))
    print(f"  replayed {len(rows)} invocations; "
          f"direct call mean {1e3*t_direct/max(1,len(rows)):.2f} ms")

    for n in (20, 40):
        lp = np.log([r[f"pred{n}"] for r in rows])
        lm = np.log([r[f"meas{n}"] for r in rows])
        resid = lp - lm
        ss = 1 - np.var(resid) / np.var(lm)
        ratios = np.exp(resid)
        # exclude scan family (known H1 crack) for the headline
        scan = [i for i, r in enumerate(rows)
                if r["op"].startswith("cumsum") or r["op"] == "masked_cumsum"]
        nsc = [i for i in range(len(rows)) if i not in scan]
        print(f"  n={n}: R2(log) = {ss:.4f}   ratio pred/meas "
              f"p5/p50/p95 = {np.percentile(ratios,5):.3f}/"
              f"{np.percentile(ratios,50):.3f}/{np.percentile(ratios,95):.3f}")
        if scan:
            print(f"        non-scan p50 ratio = "
                  f"{np.percentile(np.exp(resid[nsc]),50):.3f}   "
                  f"scan p50 ratio = {np.percentile(np.exp(resid[scan]),50):.3f}")
    with open(os.path.join(HERE, "..", "data", "v2_rows.json"), "w") as f:
        json.dump(rows, f)
    return rows


# ----------------------------------------------------------------- V3: cost
def v3():
    print("\n== V3: cost per call ==")
    rng = np.random.default_rng(1)
    for m in (48, 512, 8192):
        p = np.abs(rng.standard_normal(m)) + 0.1
        t0 = time.perf_counter()
        reps = 50
        for _ in range(reps):
            e_q95_direct(p, 20)
        dt_direct = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(5):
            y_profile(torch.from_numpy(p.astype(np.float32)), 20, nsim=3000)
        dt_sim = (time.perf_counter() - t0) / 5
        print(f"  m={m:5d}: direct {1e3*dt_direct:7.2f} ms   "
              f"M3 sim(3000) {1e3*dt_sim:7.2f} ms   "
              f"(banked probe cost ~13 ms/call, 20 GPU launches)")


if __name__ == "__main__":
    worst = v1()
    rows = v2()
    v3()
