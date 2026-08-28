"""
The attention saturation-onset law, derived and tested.

THE LAW. For softmax attention with Q,K,V ~ N(0,1) entries at (N, D), the
large_magnitude_qk variant scales Q and K by kappa, so logits are
kappa^2 x (unit-variance Gaussians). The perturbation check perturbs the
PRIMARY input only (Q) with d = randn * delta_scale * std(kappa*Q), which
perturbs row i's logits by ~iid N(0, tau_e^2), tau_e = delta_scale * kappa^2
(delta_scale = 1e-3). The inf-norm sensitivity is dominated by the row
whose top-2 logit gap g_i is smallest (response ~ e^{-g_i}); on that row the
softmax is a two-state logistic, and for g >> 1:

    s_meas ~ e^{-g} |1 - e^{-dg}| |dV|        (finite difference)
    ||Jd|| ~ e^{-g} |dg| |dV|                 (exact derivative)

    =>  r = (1 - e^{-dg}) / dg =: phi(dg),    dg ~ N(0, tau^2),
        tau = sqrt(2) * delta_scale * kappa^2

Everything cancels except dg: the per-delta ratio distribution is the
PARAMETER-FREE pushforward of N(0, tau^2) through phi, independent of the
record's own gap g (saturated regime) and of V. The old single-parameter
model phi(a), one fitted a per record, is the special case that collapses
the dg-distribution to a point -- which is why it matched rank/magnitude
but never per-record values: each delta draws its own a_k = -dg_k, so the
"paired scatter" IS the law's random variable, not noise around it.

Tests:
  T1  Per-delta: measured log r_k vs law-predicted log phi(dg_k), dg_k
      computed from the record's own dominant row -- a per-DELTA
      correlation, far stronger than a distribution match.
  T2  Pooled ratio quantiles at kappa=20 vs the analytic pushforward
      (compare_banked.py sets these against the banked GPU arm's ratios).
  T3  Per-record median distribution (the Gram flag statistic).
  T4  g-independence: record median vs record g_min uncorrelated.
  T5  The ladder-defect law: defect(dg) = |(1-e^{-dg}) - 10(1-e^{-dg/10})|
      / |1-e^{-dg}| -- should reproduce the banked 2026-08-26 defects
      (6.6-27.7%) from the same dg distribution.
  T6  kappa sweep -> where attention leaves scope and BY WHICH SCREEN:
      median_k phi(dg_k) concentrates at phi(0) = 1, so the Gram median is
      structurally blind to this mechanism; exit happens via the fp32
      floor as s ~ e^{-kappa^2 G_min} collapses below 32 ulp.

CPU emulation: fp32 torch softmax for the measured side (the GPU kernel's
online softmax is the same arithmetic within fp32 noise at these margins),
float64 torch.func.jvp of the registered math definition for the exact side
-- the same instrument scope_detect.measure_gram uses.

Run:  .venv/bin/python onset_law.py
"""
import json
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from verification.layer2_numeric_oracle import math_refs

torch.manual_seed(20260828)

N, D = 64, 32
DELTA_SCALE = 1e-3
KAPPA = 20.0
N_DELTAS = 20
OPS = ["scaled_dot_product_attention", "causal_flash_attention"]
QS = (0.05, 0.25, 0.5, 0.75, 0.95)


def phi(u):
    return 1.0 if u == 0 else (1.0 - math.exp(-u)) / u


def f32(op, q_, k_, v_):
    s = (q_ @ k_.T) / math.sqrt(q_.shape[1])
    if op == "causal_flash_attention":
        s = s.masked_fill(torch.triu(torch.ones(s.shape, dtype=torch.bool),
                                     diagonal=1), float("-inf"))
    return torch.softmax(s, dim=-1) @ v_


def ulp_at(m):
    if m == 0 or not math.isfinite(m):
        return float("nan")
    return 2.0 ** (math.floor(math.log2(abs(m))) - 23)


def run_record(op, kappa, n_deltas=N_DELTAS):
    """One synthetic invocation: per-delta (r, r_pred, dg), gram median,
    floor statistic, and the record's realized min gap."""
    Q0, K0, V = torch.randn(N, D), torch.randn(N, D), torch.randn(N, D)
    x, kk = (kappa * Q0).float(), (kappa * K0).float()

    fn = math_refs.get(op)
    f = lambda t: fn(t, kk.double(), V.double())
    base = f32(op, x, kk, V)
    x_std = float(x.std())
    u_out = ulp_at(float(base.abs().max()))

    L = (x.double() @ kk.double().T) / math.sqrt(D)
    if op == "causal_flash_attention":
        L = L.masked_fill(torch.triu(torch.ones(L.shape, dtype=torch.bool),
                                     diagonal=1), float("-inf"))
    top2 = torch.topk(L, k=2, dim=-1)
    gaps = top2.values[:, 0] - top2.values[:, 1]
    finite = torch.isfinite(gaps)
    g_min = float(gaps[finite].min()) if finite.any() else float("nan")

    deltas, s_all = [], []
    for _ in range(n_deltas):
        d = torch.randn(N, D) * DELTA_SCALE * x_std
        s_meas = float((f32(op, x + d, kk, V) - base).abs().max())
        s_all.append(s_meas)
        _, jd = torch.func.jvp(f, (x.double(),), (d.double(),))
        s_lin = float(jd.abs().max())
        if s_meas <= 0 or s_lin <= 0:
            continue
        i_star = int(jd.abs().max(dim=-1).values.argmax())
        if not bool(finite[i_star]):
            continue
        dL = (d.double() @ kk.double().T)[i_star] / math.sqrt(D)
        dg = float(dL[int(top2.indices[i_star, 0])]
                   - dL[int(top2.indices[i_star, 1])])
        deltas.append({"r": s_meas / s_lin, "r_pred": phi(dg), "dg": dg})

    med = (sorted(v["r"] for v in deltas)[len(deltas) // 2]
           if len(deltas) >= 5 else None)
    s_all.sort()
    sulp = s_all[len(s_all) // 2] / u_out if u_out == u_out else float("nan")
    return {"deltas": deltas, "median_r": med, "g_min": g_min, "sulp": sulp}


def quant(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def corr(pairs):
    n = len(pairs)
    ma = sum(a for a, _ in pairs) / n
    mb = sum(b for _, b in pairs) / n
    cov = sum((a - ma) * (b - mb) for a, b in pairs)
    sa = math.sqrt(sum((a - ma) ** 2 for a, _ in pairs))
    sb = math.sqrt(sum((b - mb) ** 2 for _, b in pairs))
    return cov / (sa * sb) if sa > 0 and sb > 0 else float("nan")


def main():
    tau = math.sqrt(2) * DELTA_SCALE * KAPPA ** 2
    out = {"tau_kappa20": tau}
    print(f"tau(kappa=20) = {tau:.4f}   [sqrt(2)*1e-3*kappa^2 -- derived, no fit]")

    NREC = 150
    ens = {op: [run_record(op, KAPPA) for _ in range(NREC)] for op in OPS}

    print("\nT1 per-delta law r_k ~ phi(dg_k):")
    for op in OPS:
        pairs = [(math.log10(v["r"]), math.log10(v["r_pred"]))
                 for rec in ens[op] for v in rec["deltas"] if v["r_pred"] > 0]
        resid = [a - b for a, b in pairs]
        mr = sum(resid) / len(resid)
        sr = math.sqrt(sum((z - mr) ** 2 for z in resid) / len(resid))
        spread = math.sqrt(sum(b * b for _, b in pairs) / len(pairs))
        print(f"  {op[:30]:30s} n={len(pairs)}  corr={corr(pairs):.3f}  "
              f"residual sd={sr:.4f} dex vs law spread {spread:.4f} dex")
        out[f"T1_{op}"] = {"n": len(pairs), "corr": corr(pairs),
                           "resid_sd": sr, "law_spread": spread}

    print("\nT2 pooled ratio quantiles at kappa=20:")
    g = torch.Generator().manual_seed(7)
    mc = [phi(float(torch.randn(1, generator=g)) * tau) for _ in range(200000)]
    for op in OPS:
        rr = [v["r"] for rec in ens[op] for v in rec["deltas"]]
        print(f"  {op[:30]:30s} " + "  ".join(
            f"P{int(p*100):02d}={quant(rr, p):.3f}" for p in QS))
        out[f"T2_{op}"] = {str(p): quant(rr, p) for p in QS}
    print(f"  {'analytic pushforward':30s} " + "  ".join(
        f"P{int(p*100):02d}={quant(mc, p):.3f}" for p in QS))
    out["T2_pushforward"] = {str(p): quant(mc, p) for p in QS}

    print("\nT3 per-record medians (the Gram flag statistic):")
    for op in OPS:
        meds = [rec["median_r"] for rec in ens[op] if rec["median_r"]]
        print(f"  {op[:30]:30s} P05={quant(meds, .05):.3f} P50={quant(meds, .5):.3f} "
              f"P95={quant(meds, .95):.3f} min={min(meds):.3f} max={max(meds):.3f}")
        out[f"T3_{op}"] = {"p05": quant(meds, .05), "p50": quant(meds, .5),
                           "p95": quant(meds, .95), "min": min(meds),
                           "max": max(meds), "n": len(meds)}

    print("\nT4 g-independence (saturated regime):")
    for op in OPS:
        pts = [(math.log10(rec["median_r"]), rec["g_min"])
               for rec in ens[op] if rec["median_r"] and rec["g_min"] > 5]
        c = corr(pts)
        print(f"  {op[:30]:30s} corr(median log r, g_min) = {c:+.3f}  "
              f"(n={len(pts)}, noise level {1.96/math.sqrt(len(pts)):.3f})")
        out[f"T4_{op}"] = {"corr": c, "n": len(pts)}

    print("\nT5 ladder-defect law from the same dg distribution:")
    g = torch.Generator().manual_seed(8)
    defs_ = []
    for _ in range(20000):
        dg = float(torch.randn(1, generator=g)) * tau
        if dg == 0:
            continue
        s1, s01 = 1 - math.exp(-dg), 1 - math.exp(-dg / 10)
        defs_.append(abs(s1 - 10 * s01) / abs(s1) * 100)
    med20 = []
    for i in range(0, 8000, 20):
        med20.append(sorted(defs_[i:i + 20])[10])
    print(f"  per-delta defect P50 = {quant(defs_, .5):.1f}%;  per-record "
          f"median-of-20: P05={quant(med20, .05):.1f}% P50={quant(med20, .5):.1f}% "
          f"P95={quant(med20, .95):.1f}%")
    print("  (banked arm-D 2026-08-26 medians on these classes: 6.6-27.7%)")
    out["T5"] = {"delta_p50": quant(defs_, .5),
                 "rec_p05": quant(med20, .05), "rec_p50": quant(med20, .5),
                 "rec_p95": quant(med20, .95)}

    print("\nT6 scope exit vs kappa (sdpa, 80 invocations each):")
    print(f"  {'kappa':>6} {'tau':>7} | {'floor%':>7} {'gram%':>7} "
          f"{'med s/ulp':>12} {'P05 s/ulp':>12}")
    sweep = {}
    for kappa in (5, 10, 15, 20, 25, 30, 35, 40, 50):
        recs = [run_record("scaled_dot_product_attention", kappa, 10)
                for _ in range(80)]
        sulps = [r["sulp"] for r in recs if r["sulp"] == r["sulp"]]
        meds = [r["median_r"] for r in recs
                if r["median_r"] and r["sulp"] >= 32]
        nf = 100 * sum(1 for s in sulps if s < 32) / len(sulps)
        ng = (100 * sum(1 for m in meds
                        if abs(math.log10(m)) >= math.log10(2)) / len(meds)
              if meds else float("nan"))
        t = math.sqrt(2) * DELTA_SCALE * kappa ** 2
        print(f"  {kappa:6.0f} {t:7.3f} | {nf:6.1f}% {ng:6.1f}% "
              f"{quant(sulps, .5):12.1f} {quant(sulps, .05):12.1f}")
        sweep[kappa] = {"floor_pct": nf, "gram_pct": ng,
                        "med_sulp": quant(sulps, .5),
                        "p05_sulp": quant(sulps, .05)}
    out["T6"] = sweep

    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "data",
                                     "onset_law.json"), "w"), indent=1)
    print("\ndone; data/onset_law.json written")


if __name__ == "__main__":
    main()
