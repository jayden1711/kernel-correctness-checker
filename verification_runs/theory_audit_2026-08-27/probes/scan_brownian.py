"""H1 — Brownian/Gram-matrix law for the scan family's adaptive tolerance.

Claim under test: for an exactly-linear operator the dimensionless tolerance
    y = adaptive_tol / (3 sigma L)
is a functional of the Jacobian's GRAM MATRIX J J^T alone (not just the
row-norm profile M3 uses). For prefix scans the Gram matrix is the Brownian
covariance min(i,j)+1, so the parent of the sensitivity samples is the maximum
of |Brownian motion| sampled on a grid -- and the reflection principle gives a
closed form. The +24.7% M3 over-prediction on scans should be EXACTLY the
independent-coordinates-vs-Brownian gap.

Everything here is computed from the known Jacobian structure only. No number
is fitted to the banked GPU data; the banked data is used solely as the
measurement the predictions are compared against.

CPU-only. Banked inputs:
  ../../phase1_derivations_2026-08-27/native_run/phase1_native.jsonl  (24 scan recs)
  ../../phase1_derivations_2026-08-27/native_run/pass2.jsonl          (adv variants)
"""

import json
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.join(HERE, "../../phase1_derivations_2026-08-27/native_run")

R, C = 64, 512          # valid_shapes[0] for all four scan specs
NS = 40                 # perturbation samples in the native run
NREP = 400              # replications of the 40-sample estimator per op
SEED = 0

rng = np.random.default_rng(SEED)


def q95_torch(v):
    """torch.quantile(v, 0.95) with linear interpolation, replicated exactly."""
    v = np.sort(v, axis=-1)
    n = v.shape[-1]
    h = 0.95 * (n - 1)
    lo = int(math.floor(h))
    frac = h - lo
    return v[..., lo] * (1 - frac) + v[..., lo + 1] * frac


# ---------------------------------------------------------------- banked data
def load_measured():
    recs = [json.loads(l) for l in open(os.path.join(NATIVE, "phase1_native.jsonl"))]
    out = {}
    for r in recs:
        if "cumsum" in r["op"]:
            y = r["tol"] / (3 * r["sigma"] * r["L_closed"])
            out.setdefault(r["op"], []).append(
                dict(y=y, y_M3=r["y_M3"], L_closed=r["L_closed"]))
    return out


def load_adv():
    recs = [json.loads(l) for l in open(os.path.join(NATIVE, "pass2.jsonl"))]
    return [r for r in recs if "cumsum" in r["op"] and r.get("kind") == "adv"]


# ------------------------------------------------- exact-law parent samplers
def parent_samples(op, n_draws, ncols=C, chunk=2000):
    """n_draws iid samples of s/sigma = ||J g||_inf for the op's exact J.

    Uses only the known Jacobian structure. For masked_cumsum a fresh
    Bernoulli(0.5) mask is drawn per sample (the spec's own distribution);
    the returned samples are already divided by that sample's own closed-form
    L, matching how the banked y divides by the invocation's own L_closed.
    Others are divided by their closed-form L afterwards by the caller.
    """
    outs = []
    done = 0
    while done < n_draws:
        b = min(chunk, n_draws - done)
        g = rng.standard_normal((b, R, ncols), dtype=np.float32)
        if op == "cumsum":
            w = np.cumsum(g, axis=-1)
            s = np.abs(w).max(axis=(1, 2)) / math.sqrt(ncols)
        elif op == "cumsum_reverse":
            w = np.cumsum(g[..., ::-1], axis=-1)
            s = np.abs(w).max(axis=(1, 2)) / math.sqrt(ncols)
        elif op == "cumsum_exclusive":
            w = np.cumsum(g, axis=-1)[..., :-1]     # exclusive: W_0=0 dropped, max over W_1..W_{C-1}
            s = np.abs(w).max(axis=(1, 2)) / math.sqrt(ncols - 1)
        elif op == "masked_cumsum":
            m = (rng.random((b, R, ncols)) < 0.5).astype(np.float32)
            w = np.cumsum(m * g, axis=-1)
            L = np.sqrt((m ** 2).sum(axis=-1).max(axis=-1))   # closed form, per sample's mask
            s = np.abs(w).max(axis=(1, 2)) / L
        elif op == "m3_cumsum":
            # M3's orthogonal-rows assumption: independent z_i weighted by ||J_i||/L
            wgt = np.sqrt((np.arange(ncols, dtype=np.float32) + 1) / ncols)
            s = (np.abs(g) * wgt).max(axis=(1, 2))
        else:
            raise ValueError(op)
        outs.append(s.astype(np.float64))
        done += b
    return np.concatenate(outs)


def estimator_mean(op, ncols=C, nrep=NREP):
    """Mean and sd of y-hat = q95_40(parent)/1 over nrep replications."""
    s = parent_samples(op, nrep * NS, ncols=ncols)
    y = q95_torch(s.reshape(nrep, NS))
    return y.mean(), y.std(ddof=1)


# ------------------------------------------------------- reflection principle
def F_maxabs_bm(a, terms=200):
    """P( max_{0<=t<=1} |B_t| <= a ), reflection-principle theta series."""
    if a <= 0:
        return 0.0
    tot = 0.0
    for j in range(terms):
        k = 2 * j + 1
        tot += ((-1) ** j / k) * math.exp(-k * k * math.pi ** 2 / (8 * a * a))
    return max(0.0, min(1.0, 4.0 / math.pi * tot))


def closed_form_y(nrows=R, n=NS, grid=8001, amax=6.0):
    """E[q95_n] of max over nrows iid copies of max|B| on [0,1].

    Exact numeric integration of the order-statistic blend the shipped
    torch.quantile computes: h=0.95(n-1), E[(1-frac) X_(lo+1:n) + frac X_(lo+2:n)].
    Continuous Brownian limit -- carries NO discrete-walk correction.
    """
    a = np.linspace(1e-6, amax, grid)
    F1 = np.array([F_maxabs_bm(x) for x in a])
    F = F1 ** nrows                       # row-max of R iid copies
    h = 0.95 * (n - 1)
    lo = int(math.floor(h))
    frac = h - lo

    def e_order(j):                        # E[X_(j:n)], j 1-indexed
        # F_(j)(x) = P(at least j of n below x) = sum_{i>=j} C(n,i) F^i (1-F)^(n-i)
        Fj = np.zeros_like(F)
        for i in range(j, n + 1):
            Fj += math.comb(n, i) * F ** i * (1 - F) ** (n - i)
        # X >= 0, so E[X_(j:n)] = int_0^inf (1 - F_(j)) dx
        return np.trapezoid(1 - Fj, a)

    return (1 - frac) * e_order(lo + 1) + frac * e_order(lo + 2)


# ---------------------------------------------------------------------- main
def main():
    meas = load_measured()
    print("=== banked GPU measurements (y = tol / 3 sigma L_closed) ===")
    for op, rows in meas.items():
        ys = np.array([r["y"] for r in rows])
        m3 = np.array([r["y_M3"] for r in rows])
        print(f"{op:18s} y_meas mean {ys.mean():.4f} sd {ys.std(ddof=1):.4f} "
              f"(n={len(ys)})   y_M3 mean {m3.mean():.4f}")

    print("\n=== exact-law (Gram-matrix) prediction, no fitted constants ===")
    preds = {}
    for op in ["cumsum", "cumsum_reverse", "cumsum_exclusive", "masked_cumsum"]:
        mu, sd = estimator_mean(op)
        preds[op] = (mu, sd)
        ys = np.array([r["y"] for r in meas[op]])
        sem = sd / math.sqrt(len(ys))     # predicted sd of a 6-invocation mean
        z = (ys.mean() - mu) / (ys.std(ddof=1) / math.sqrt(len(ys)))
        print(f"{op:18s} y_pred {mu:.4f} +- {sd:.4f}   meas/pred "
              f"{ys.mean()/mu:.4f}   z(meas mean vs pred) {z:+.2f}")

    print("\n=== M3-orthogonal baseline (should reproduce banked y_M3) ===")
    mu3, sd3 = estimator_mean("m3_cumsum")
    m3_banked = np.array([r["y_M3"] for r in meas["cumsum"]]).mean()
    print(f"m3(cumsum) sim {mu3:.4f} +- {sd3:.4f}   banked y_M3 {m3_banked:.4f} "
          f"ratio {m3_banked/mu3:.4f}")
    print(f"independence-vs-Brownian gap: {mu3/preds['cumsum'][0]:.4f} "
          f"(measured family M3 residual was ~1.25)")

    print("\n=== falsification A: out-of-sample shape C=333 (non_power_of_two) ===")
    mu333, sd333 = estimator_mean("cumsum", ncols=333)
    adv = load_adv()
    for r in adv:
        if r["op"] == "cumsum" and r.get("variant") == "non_power_of_two":
            ym = r["tol"] / (3 * r["sigma"] * math.sqrt(333))
            print(f"measured y {ym:.4f}   predicted {mu333:.4f} +- {sd333:.4f} "
                  f"  z {(ym-mu333)/sd333:+.2f}")

    print("\n=== falsification B: input-invariance of y across adversarial variants ===")
    for op in ["cumsum", "cumsum_reverse", "cumsum_exclusive"]:
        rows = [r for r in adv if r["op"] == op and r["m"] == R * C]
        Lc = math.sqrt(C) if op != "cumsum_exclusive" else math.sqrt(C - 1)
        ys = [(r["variant"], r["tol"] / (3 * r["sigma"] * Lc)) for r in rows]
        print(op, " ".join(f"{v}={y:.4f}" for v, y in ys))

    print("\n=== closed form (reflection principle, continuous limit) ===")
    try:
        ycf = closed_form_y()
        print(f"closed-form y (C -> inf) = {ycf:.4f}")
        print(f"vs exact-law sim at C=512: {preds['cumsum'][0]:.4f} "
              f"(gap = discrete-walk correction)")
        # discretization ladder: parent median at several C
        print("\ndiscrete-walk convergence (estimator mean vs C):")
        for nc in [64, 128, 512, 2048]:
            mu, sd = estimator_mean("cumsum", ncols=nc, nrep=200)
            print(f"  C={nc:5d}  y_pred {mu:.4f}  gap to continuous {ycf-mu:+.4f} "
                  f" gap*sqrt(C) {(ycf-mu)*math.sqrt(nc):+.3f}")
    except Exception as e:
        print("scipy unavailable:", e)

    json.dump({k: list(v) for k, v in preds.items()},
              open(os.path.join(HERE, "../data/scan_brownian_preds.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
