"""
B.3 CHAINING -- probe-free prediction of the whole n_samples curve.

The claim under test (generalization/FINDINGS.md B.3, flagged "not validated
end-to-end"; structural_l.py y_profile docstring, "a default-configured run is
simulating an order statistic one step outside what was checked"): M3's
structural parent distribution, chained with the exact effective-quantile
identity p_eff(n) = (0.95n + 0.05)/(n+1), predicts tol_n/tol_40 with NO
probing of the reference -- in particular at the shipped default n = 20,
which was never directly validated (M3 was validated at n = 40 only).

Three predictors per invocation, all compared against the measured prefix
curve from the banked native 40-sample sensitivity vectors
(../../adaptive_tol_theory_2026-08-25/native_run/gpu_native.jsonl):

  DIRECT   E[q95_n]/E[q95_40] under the structural M3 parent
           s = max_i w_i |z_i|, w = row-norm profile / L, computed EXACTLY:
           the parent CDF is F(t) = prod_i (2 Phi(t / w_i) - 1), evaluated on
           a grid, sampled by inverse transform. Zero fitted constants, no
           kernel, no probe.
  CHAIN    the validated one-parameter Gumbel model
           tol_n/tol_40 = [1 + rho G(p_eff(n))]/[1 + rho G(p_eff(40))]
           with rho taken from the STRUCTURAL CV (from F's exact moments)
           instead of the measured sample CV -- the literal B.3 composition.
  GUMBEL-M the same model with the MEASURED per-invocation CV (the already
           validated route, re-run here as the baseline the chain must match).

Inputs are replayed bit-for-bit (np.random.default_rng(0), tritonbench
registry order, 6 draws per entry, argmax/argmin consume draws before their
exclusion -- exactly gpu_native.py's loop). Replay is verified per row by
sigma_banked == 1e-3 * std(x) before anything else is computed.

Profiles come from structural_l.row_norms -- the shipped closed forms.

Scan-family arm (phase1_native.jsonl, 24 invocations at (64, 512)): the scan
profile is shape-only, and the M3 independence assumption is known-wrong
there (+24.7%, theory_audit H1). Both parents are run:
  M3 independent-max over nu_i = sqrt(i+1), and
  the EXACT Brownian parent (max_k |W_k| per row, row-max over 64 iid rows),
to test whether the n-CURVE (a shape ratio) inherits the correlation error
that the LEVEL of y does, and whether the exact Gram parent repairs it.

Writes data/b3_chain.json + a human-readable log.
"""

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "autokernel", "files"))

from tritonbench_registry import OPS, FAMILIES  # numpy/torch only at import
from verification.layer2_numeric_oracle.structural_l import row_norms

NATIVE = os.path.join(HERE, "..", "..", "adaptive_tol_theory_2026-08-25",
                      "native_run", "gpu_native.jsonl")
PHASE1 = os.path.join(HERE, "..", "..", "phase1_derivations_2026-08-27",
                      "native_run", "phase1_native.jsonl")
DATA = os.path.join(HERE, "..", "data")

EXCLUDE = {"argmax", "argmin"}
N_GRID = [2, 3, 5, 10, 15, 20, 30, 40]
NREP = 4000          # replications of the 40-draw estimator per invocation
NS = 40
GAMMA = 0.5772156649015329


def qlin(xs, q=0.95):
    s = sorted(xs)
    n = len(s)
    h = q * (n - 1)
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def p_eff(n):
    return (0.95 * n + 0.05) / (n + 1)


def G(p):
    return -math.log(-math.log(p))


def gumbel_curve(cv):
    """rho from CV via CV = (pi/sqrt6) rho / (1 + gamma rho), then the model."""
    a = math.pi / math.sqrt(6.0)
    denom = a - GAMMA * cv
    if denom <= 0:
        return None
    rho = cv / denom
    base = 1 + rho * G(p_eff(40))
    return {n: (1 + rho * G(p_eff(n))) / base for n in N_GRID}


def parent_from_profile(w):
    """Exact CDF F(t) = prod_i (2 Phi(t/w_i) - 1) on a grid; returns
    (t_grid, F). w normalized to max 1, zeros dropped."""
    w = w[w > 0]
    w = w / w.max()
    m = w.size
    t_hi = math.sqrt(2 * math.log(max(2 * m, 4))) + 6.0
    t = np.linspace(0.0, t_hi, 4096)
    # log F = sum_i log(2 Phi(t/w_i) - 1); torch.erf is the vectorized route
    # (2 Phi(x) - 1 = erf(x/sqrt2)); chunked over i to bound memory.
    tt = torch.from_numpy(t)
    logF = torch.zeros_like(tt)
    for i0 in range(0, m, 2048):
        wi = torch.from_numpy(w[i0:i0 + 2048]).unsqueeze(1)
        c = torch.erf(tt.unsqueeze(0) / wi / math.sqrt(2.0))
        logF += torch.log(c.clamp_min(1e-300)).sum(dim=0)
    return t, torch.exp(logF).numpy()


def curve_from_cdf(t, F, rng):
    """Sample NREP x NS draws by inverse transform; return (curve, cv)."""
    u = rng.random((NREP, NS))
    s = np.interp(u, F, t)
    curves = {}
    for n in N_GRID:
        # q95 over the first n of each replication (prefix convention)
        sub = np.sort(s[:, :n], axis=1)
        h = 0.95 * (n - 1)
        lo = int(math.floor(h))
        hi = min(lo + 1, n - 1)
        q = sub[:, lo] + (h - lo) * (sub[:, hi] - sub[:, lo])
        curves[n] = float(q.mean())
    base = curves[40]
    curve = {n: curves[n] / base for n in N_GRID}
    # exact-ish moments from the samples (plenty for CV)
    cv = float(s.std(ddof=1) / s.mean())
    return curve, cv


def replay_inputs():
    """(entry_idx, inv) -> (op, np_args); exact gpu_native.py stream."""
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


def measured_curve(sens):
    base = qlin(sens)
    if base <= 0:
        return None
    return {n: qlin(sens[:n]) / base for n in N_GRID}


def main():
    os.makedirs(DATA, exist_ok=True)
    inputs = replay_inputs()
    rows = [json.loads(l) for l in open(NATIVE)]
    rows = [r for r in rows if r.get("kind") == "primary" and r.get("sens")]

    out_rows = []
    n_align = 0
    parent_cache = {}
    rng = np.random.default_rng(20260828)
    for r in rows:
        key = (int(r["entry"]), int(r["inv"]))
        op, np_args = inputs[key]
        assert op == r["op"], (key, op, r["op"])
        x, rest = to_torch64(np_args)
        sig = 1e-3 * float(x.to(torch.float32).std())
        assert abs(sig - r["sigma"]) / r["sigma"] < 1e-5, (key, sig, r["sigma"])
        n_align += 1

        rn = row_norms(op, x, tuple(rest))
        if rn is None:
            out_rows.append(dict(op=op, entry=key[0], inv=key[1],
                                 skipped="no closed form"))
            continue
        w = rn.detach().cpu().numpy().astype(np.float64)
        # cache the parent per profile fingerprint (many invocations share
        # near-identical profiles only for shape-only ops; hash exact values)
        fp = (op, w.size, round(float(w.max()), 12), round(float(w.sum()), 9))
        if fp in parent_cache:
            direct, cv_struct = parent_cache[fp]
        else:
            t, F = parent_from_profile(w)
            direct, cv_struct = curve_from_cdf(t, F, rng)
            parent_cache[fp] = (direct, cv_struct)

        meas = measured_curve(r["sens"])
        chain = gumbel_curve(cv_struct)
        gumb_m = gumbel_curve(r["cv"]) if r.get("cv") else None

        out_rows.append(dict(op=op, entry=key[0], inv=key[1],
                             m=int(r["m"]), cv_meas=r.get("cv"),
                             cv_struct=cv_struct,
                             floor_pinned=(r["tol"] <= 1.0000001e-6),
                             meas=meas, direct=direct, chain=chain,
                             gumbel_meas=gumb_m))
        print(f'{op:28s} e{key[0]:02d}i{key[1]} cv_meas={r.get("cv") or 0:.3f} '
              f'cv_struct={cv_struct:.3f} '
              f'meas20={meas[20] if meas else 0:.4f} '
              f'direct20={direct[20]:.4f}', flush=True)

    print(f"\nalignment: {n_align}/{len(rows)} rows replayed bit-consistent")

    # ------------------------------------------------------------- scans ---
    # Shape-only parents at (R, C) = (64, 512), from phase1 bank.
    p1 = [json.loads(l) for l in open(PHASE1)]
    scans = [r for r in p1 if r["op"] in
             ("cumsum", "cumsum_reverse", "cumsum_exclusive", "masked_cumsum")
             and r.get("sens")]
    R, C = 64, 512
    # M3 independent parent: nu_i = sqrt(prefix length), all rows concatenated
    w_scan = np.sqrt(np.tile(np.arange(1, C + 1), R).astype(np.float64))
    t, F = parent_from_profile(w_scan)
    m3_scan, cv_m3_scan = curve_from_cdf(t, F, rng)
    # Exact Brownian parent: per-row max |walk|, row-max over R iid rows.
    B0 = 200000
    walks = np.cumsum(np.random.default_rng(7).standard_normal((B0, C)),
                      axis=1)
    row_max = np.abs(walks).max(axis=1) / math.sqrt(C)
    ts = np.sort(row_max)
    Fr = (np.arange(1, B0 + 1) - 0.5) / B0
    Fex = Fr ** R                       # row-max over R iid rows
    brown_scan, cv_brown = curve_from_cdf(ts, Fex, rng)

    scan_rows = []
    for r in scans:
        meas = measured_curve(r["sens"])
        scan_rows.append(dict(op=r["op"], inv=r["inv"], cv_meas=r.get("cv"),
                              meas=meas))
    print(f"\nscan family: {len(scan_rows)} invocations; "
          f"cv_struct M3={cv_m3_scan:.3f} Brownian={cv_brown:.3f}, "
          f"measured cv range "
          f"{min(x['cv_meas'] for x in scan_rows):.3f}-"
          f"{max(x['cv_meas'] for x in scan_rows):.3f}")

    json.dump(dict(n_grid=N_GRID, rows=out_rows,
                   scan=dict(rows=scan_rows, m3=m3_scan,
                             m3_cv=cv_m3_scan,
                             brownian=brown_scan, brownian_cv=cv_brown,
                             m3_chain=gumbel_curve(cv_m3_scan),
                             brownian_chain=gumbel_curve(cv_brown))),
              open(os.path.join(DATA, "b3_chain.json"), "w"))
    print("wrote data/b3_chain.json")


if __name__ == "__main__":
    main()
