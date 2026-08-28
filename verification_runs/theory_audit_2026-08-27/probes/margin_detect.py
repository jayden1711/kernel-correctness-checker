"""H2 — detection probability as a function of mutant margin.

The open question from n_samples_curve_2026-08-25: is there a formula relating
a mutant's distance from correct behaviour to its detection probability?

Derivation (no new assumptions beyond the round's verified A1-A3):
  verdict = fail  <=>  max_err > max(3 q95_n(s), 1e-6)
  max_err is deterministic; the randomness is the n perturbation draws.
  With u = F(max_err / 3) the parent CDF at the implied threshold, and
  q95_n in [X_(n-1:n), X_(n:n)] (the shipped torch.quantile blend),

      u^n  <=  P(detect)  <=  u^n + n u^(n-1) (1 - u)          [exact bracket]

  and the Gumbel-tail parent of the earlier round (one parameter, from the
  sample CV) turns u into a closed form of the margin:
      u = exp(-exp(-(max_err/3 - a)/b)),  a,b from mean/sd of s.
  (Floor: if max_err <= 1e-6 then P(detect) = 0 exactly.)

Validation against the real 854 banked 40-sample GPU vectors:
  P1  transition-zone sweep: empirical P(detect) by resampling n-subsets of
      the 40 recorded sensitivities vs the bracket midpoint and Gumbel form.
  P2  split-half, non-circular: Gumbel params from samples 0..19 only,
      resampling from disjoint samples 20..39 only.
  P3  the corpus-saturation statement, now derived instead of observed:
      per-invocation P(miss) bounds for all live mutant invocations and the
      P(FP) bound for the single live reference invocation.
"""

import gzip
import json
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CURVE = os.path.join(HERE, "../../n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz")
FLOOR = 1e-6
GAMMA = 0.5772156649015329

rng = np.random.default_rng(1)


def q95_np(v):
    v = np.sort(v, axis=-1)
    n = v.shape[-1]
    h = 0.95 * (n - 1)
    lo = int(math.floor(h))
    fr = h - lo
    return v[..., lo] * (1 - fr) + v[..., min(lo + 1, n - 1)] * fr


def load():
    d = json.load(gzip.open(CURVE))
    out = []
    for ent in d["entries"]:
        sides = [("mutant", ent["mutant"])] + [("ref", r) for r in ent["refs"]]
        for kind, side in sides:
            for r in side["records"]:
                for sc in (r.get("subchecks") or []):
                    if sc.get("kind") == "perturbation_sensitivities":
                        out.append(dict(op=ent["op"], kind=kind, check=r["name"],
                                        sens=np.array(sc["sensitivities"]),
                                        max_err=sc["max_err"]))
    return out


def bracket(u, n):
    lo = u ** n
    hi = u ** n + n * u ** (n - 1) * (1 - u)
    return lo, hi


def gumbel_params(s):
    m, sd = s.mean(), s.std(ddof=1)
    if sd == 0 or m == 0:
        return None
    b = sd * math.sqrt(6) / math.pi
    a = m - GAMMA * b
    return a, b


def gumbel_u(t, a, b):
    return math.exp(-math.exp(-max(min((t - a) / b, 700), -700)))


def resample_p(s, t, n, m=4000):
    """empirical P(detect at threshold t) over m random n-subsets of s."""
    idx = np.argsort(rng.random((m, len(s))), axis=1)[:, :n]
    q = q95_np(s[idx])
    tol = np.maximum(3 * q, FLOOR)
    return (t > tol).mean()


def main():
    recs = load()
    usable = [r for r in recs if r["sens"].std() > 0 and r["sens"].mean() > 0]
    print(f"{len(recs)} invocations loaded, {len(usable)} with non-degenerate sens")

    # ---------------- P1: transition-zone sweep, plug-in + Gumbel
    sample = usable[:: max(1, len(usable) // 120)]
    errs_mid, errs_gum, rows = [], [], 0
    for r in sample:
        s = r["sens"]
        gp = gumbel_params(s)
        # thresholds spanning the transition: parent quantiles 0.5 .. 1.3*max
        for frac in [0.70, 0.85, 0.95, 1.0, 1.05, 1.15, 1.35]:
            t = 3 * np.quantile(s, 0.95) * frac
            if t <= FLOOR:
                continue
            p_emp = resample_p(s, t, 20)
            u_emp = (s <= t / 3).mean()
            lo, hi = bracket(u_emp, 20)
            errs_mid.append(abs((lo + hi) / 2 - p_emp))
            if gp:
                ug = gumbel_u(t / 3, *gp)
                lg, hg = bracket(ug, 20)
                errs_gum.append(abs((lg + hg) / 2 - p_emp))
            rows += 1
    errs_mid, errs_gum = np.array(errs_mid), np.array(errs_gum)
    print(f"\nP1 sweep: {rows} (invocation, threshold) points, n=20")
    print(f"  |bracket midpoint - resampled P|: median {np.median(errs_mid):.4f} "
          f" p90 {np.quantile(errs_mid, .9):.4f}  max {errs_mid.max():.4f}")
    print(f"  |Gumbel closed form - resampled P|: median {np.median(errs_gum):.4f} "
          f" p90 {np.quantile(errs_gum, .9):.4f}  max {errs_gum.max():.4f}")

    # n-dependence
    for n in [5, 10, 20]:
        e = []
        for r in sample[:40]:
            s = r["sens"]
            for frac in [0.85, 1.0, 1.15]:
                t = 3 * np.quantile(s, 0.95) * frac
                if t <= FLOOR:
                    continue
                p_emp = resample_p(s, t, n)
                u = (s <= t / 3).mean()
                lo, hi = bracket(u, n)
                e.append(abs((lo + hi) / 2 - p_emp))
        print(f"  n={n:2d}: median |mid-P| {np.median(e):.4f}  max {np.max(e):.4f}")

    # ---------------- P2: split-half, non-circular
    print("\nP2 split-half (params from first 20 samples, resampling from last 20):")
    e2 = []
    for r in sample:
        s1, s2 = r["sens"][:20], r["sens"][20:]
        gp = gumbel_params(s1)
        if not gp:
            continue
        for frac in [0.85, 1.0, 1.15]:
            t = 3 * np.quantile(s1, 0.95) * frac
            if t <= FLOOR:
                continue
            p_emp = resample_p(s2, t, 10)
            ug = gumbel_u(t / 3, *gp)
            lo, hi = bracket(ug, 10)
            e2.append(abs((lo + hi) / 2 - p_emp))
    e2 = np.array(e2)
    print(f"  {len(e2)} points, n=10: median {np.median(e2):.4f} "
          f" p90 {np.quantile(e2, .9):.4f}  max {e2.max():.4f}")

    # ---------------- P3: the saturation statement, derived
    print("\nP3 corpus saturation, derived from the formula:")
    live_mut = [r for r in recs if r["kind"] == "mutant" and r["max_err"] > 0]
    live_ref = [r for r in recs if r["kind"] == "ref" and r["max_err"] > 0]
    worst_miss = 0.0
    n_boundary = 0
    for r in live_mut:
        s = r["sens"]
        t = r["max_err"]
        tol40 = max(3 * np.quantile(s, 0.95), FLOOR)
        if t <= tol40:      # not detected even at n=40 -- not a caught invocation
            continue
        gp = gumbel_params(s)
        if gp is None:      # degenerate (all-equal) sens: tol is deterministic
            continue
        u = gumbel_u(t / 3, *gp)
        p_miss_hi = 1 - u ** 20         # 1 - lower bracket = miss upper bound
        worst_miss = max(worst_miss, p_miss_hi)
        if p_miss_hi > 1e-6:
            n_boundary += 1
            print(f"  near-boundary mutant: {r['op']}/{r['check']} "
                  f"margin {t/tol40:.2f}x  P(miss) <= {p_miss_hi:.3e}")
    print(f"  detected mutant invocations: worst P(miss) bound = {worst_miss:.3e} "
          f"({n_boundary} above 1e-6)")
    for r in live_ref:
        s = r["sens"]
        t = r["max_err"]
        gp = gumbel_params(s)
        u = gumbel_u(t / 3, *gp) if gp else float(t / 3 >= s.max())
        lo, hi = bracket(u, 20)
        tol_med = max(3 * np.quantile(s, 0.95), FLOOR)
        print(f"  live reference: {r['op']}/{r['check']} margin {t/tol_med:.2e} "
              f" P(FP) <= {hi:.3e}")


if __name__ == "__main__":
    main()
