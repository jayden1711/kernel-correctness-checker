"""
The response-curve law: P(catch | design margin) is a derived functional of
the structural parent -- the checker's ROC against bug magnitude, predicted
with no GPU and no fit.

THE LAW. For an m-series mutant (epilogue mis-scale DELTA = m * rho0, rho0
the design median of tol/max|f|), the perturbation check catches iff

    DELTA * M(x)  >  tol(x, z) = max(3 * q95_20(s), 1e-6),

where over a fresh harness input x and fresh perturbation draws z:
  * M(x) = max|f(x)| -- an extreme-value functional of the output field,
  * s_k  = sigma(x) * max_i w_i(x) |z_ik| -- the structural parent
    (closed-form row norms; the same object H1/Gram derived),
  * q95_20 = torch.quantile's interpolated 95th of 20 draws.

Everything on the right has a distribution DERIVED from the operator's
structure; nothing is fitted. The predicted response curve is

    P_catch(m) = P[ m * rho0 * M(x) > tol(x, z) ]

evaluated by Monte Carlo OF THE LAW (fresh CPU input draws, parent
inverse-transform draws; deterministic seed) -- the same operative form the
attention onset law used for its record-level statistic.

VALIDATION (attempted falsification):
  T1  25 (op, margin) points, 10 GPU seeds each, banked in
      ../near_miss_2026-08-28/data/near_miss_gpu.json: predicted P vs
      observed k/10, scored by exact binomial two-sided tail probability.
  T2  the per-op realized-margin distribution: predicted (median, CV)
      vs banked realized margins per op.
  T3  the pooled response curve at the five design margins vs the banked
      0/6/42/90/100 %.
  T4  the v-series straddle widths for the three ops whose binding check
      is NOT floor-adjacent (layernorm affine, l2norm cross_shape,
      sum_reduction cross_shape): predicted within-mutant catch fraction
      at each design margin vs the banked v-series verdict counts.
      gelu/softmax are EXCLUDED BY DERIVATION: their binding tolerance is
      the 1e-6 floor +- fp-quantization noise, i.e. the s/ulp < 32 regime
      the scope round proved outside the parent's validity domain. The
      exclusion is stated, not silent.

Run:  .venv/bin/python response_law.py
"""
import importlib
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verification_runs",
                                "direct_tol_2026-08-28", "probes"))
sys.path.insert(0, os.path.join(ROOT, "verification_runs",
                                "binding_law_2026-08-28", "probes"))
DATA = os.path.join(HERE, "..", "data")

from verification.layer2_numeric_oracle.structural_l import row_norms
from predict_bindings import (ln_ref, softmax_ref, gelu_ref, l2norm_ref,
                              sum_ref, REFS)

SHAPE = (64, 128)
OPS = ["layernorm", "softmax", "gelu", "l2norm", "sum_reduction"]
MARGINS = [0.5, 0.8, 1.0, 1.25, 2.0]
MKEYS = ["m050", "m080", "m100", "m125", "m200"]
NREP = 3000
SQRT2 = math.sqrt(2.0)


def parent_cdf_grid(rn):
    """(t, F) for the ABSOLUTE parent s = max_i rn_i |z_i| (rn absolute)."""
    rn = rn[rn > 0]
    L = rn.max()
    w = rn / L
    w = w[w >= 0.25]                      # validated truncation
    m = w.size
    t_hi = (math.sqrt(2 * math.log(max(2 * m, 4))) + 6.0)
    t = np.linspace(0, t_hi, 1024)
    a = t[None, :] / (SQRT2 * w[:, None])
    c = np.clip(torch.erf(torch.from_numpy(a)).numpy(), 1e-300, 1.0)
    F = np.exp(np.log(c).sum(axis=0))
    return t * L, F


def qlin20(v):
    s = np.sort(v, axis=-1)
    return s[..., 18] * 0.95 + s[..., 19] * 0.05


def ensemble(op, nrep=NREP, seed=12345):
    """Fresh-input law ensemble: per replicate (M, tol)."""
    spec = importlib.import_module(f"verification.specs.{op}").get_spec()
    ref = REFS[op]
    g = np.random.default_rng(seed)
    Ms, tols = [], []
    for i in range(nrep):
        torch.manual_seed(int(g.integers(2**31)))
        inputs = spec.make_inputs(SHAPE, "cpu", torch.float32)
        if isinstance(inputs, tuple):
            x, comps = inputs[0], list(inputs[1:])
            f = spec.run_reference(ref, inputs)
        else:
            x, comps = inputs, []
            f = ref(inputs)
        M = float(f.abs().max())
        x_std = float(x.float().std()) or 1.0
        sigma = 1e-3 * x_std
        rn = row_norms(op, x, comps).double().numpy().ravel()
        t, F = parent_cdf_grid(rn)
        u = g.random(20)
        s = sigma * np.interp(u, F, t)
        tol = max(3.0 * qlin20(s), 1e-6)
        Ms.append(M)
        tols.append(tol)
    return np.array(Ms), np.array(tols)


def main():
    os.makedirs(DATA, exist_ok=True)
    torch.set_num_threads(4)
    bank = [r for r in json.load(open(os.path.join(
        ROOT, "verification_runs", "near_miss_2026-08-28", "data",
        "near_miss_gpu.json")))["records"] if "margin" in r]
    design = json.load(open(os.path.join(
        ROOT, "verification_runs", "near_miss_2026-08-28", "data",
        "design_deltas.json")))

    results = {}
    print("== T1/T2/T3: m-series response curves ==")
    pooled_pred = {k: [] for k in MKEYS}
    pooled_obs = {k: [0, 0] for k in MKEYS}
    for op in OPS:
        Ms, tols = ensemble(op)
        rho0 = design[op]["rho_median"]
        results[op] = {}
        # T2: realized margin distribution at design margin 1.0
        marg = rho0 * Ms / tols
        pm = np.percentile(marg, [5, 50, 95])
        cv = marg.std() / marg.mean()
        obs_m = [r["margin"] for r in bank if r["op"] == op
                 and r["mutant"] == "m100"]
        print(f"  {op:14s} margin@m100: predicted p5/p50/p95 = "
              f"{pm[0]:.3f}/{pm[1]:.3f}/{pm[2]:.3f} (CV {100*cv:.1f}%)   "
              f"banked 10-seed min/med/max = {min(obs_m):.3f}/"
              f"{np.median(obs_m):.3f}/{max(obs_m):.3f}")
        results[op]["pred_margin_p5_p50_p95"] = list(pm)
        results[op]["pred_margin_cv"] = float(cv)
        results[op]["obs_margins_m100"] = obs_m
        # T1: per-margin catch probabilities
        for mk, m in zip(MKEYS, MARGINS):
            delta = design[op]["deltas"][str(m)]
            p = float(np.mean(delta * Ms > tols))
            obs = [not r["pert_passed"] for r in bank
                   if r["op"] == op and r["mutant"] == mk]
            k, n = sum(obs), len(obs)
            # exact binomial two-sided tail prob of k given p
            from math import comb
            pmf = [comb(n, j) * p**j * (1 - p)**(n - j) for j in range(n + 1)]
            tail = sum(q for q in pmf if q <= pmf[k] + 1e-15)
            flag = "OK " if tail > 0.05 else ("~  " if tail > 0.01 else "REJ")
            print(f"      {mk}: predicted P = {p:5.1%}   observed {k}/{n}"
                  f"   binom tail p = {tail:.3f} {flag}")
            results[op][mk] = {"pred_p": p, "obs_k": k, "obs_n": n,
                               "tail": tail}
            pooled_pred[mk].append(p)
            pooled_obs[mk][0] += k
            pooled_obs[mk][1] += n
    print("\n  T3 pooled curve:")
    for mk in MKEYS:
        pp = np.mean(pooled_pred[mk])
        k, n = pooled_obs[mk]
        print(f"      {mk}: predicted {pp:5.1%}   observed {k}/{n} "
              f"= {k/n:5.1%}")

    # ---- T4: v-series straddle for the non-floor bindings ----------------
    print("\n== T4: v-series straddles (non-floor bindings only) ==")
    vbank = json.load(open(os.path.join(
        ROOT, "verification_runs", "near_miss_verdict_2026-08-28", "data",
        "v_series_gpu.json")))["records"]
    vdesign = json.load(open(os.path.join(
        ROOT, "verification_runs", "near_miss_verdict_2026-08-28", "data",
        "design_verdict.json")))
    g = np.random.default_rng(999)
    VOPS = {"layernorm": "affine", "l2norm": "cross_shape",
            "sum_reduction": "cross_shape"}
    vres = {}
    for op, kind in VOPS.items():
        dstar = vdesign[op]["binding"][1]
        spec = importlib.import_module(f"verification.specs.{op}").get_spec()
        ref = REFS[op]
        # predicted catch probability of the BINDING COMPARATOR at each
        # v-margin, over fresh input draws (exact comparator emulation)
        fracs = {}
        for vm, mult in [("v050", .5), ("v080", .8), ("v100", 1.),
                         ("v125", 1.25), ("v200", 2.)]:
            delta = dstar * mult
            catches = 0
            nrep = 400
            for i in range(nrep):
                torch.manual_seed(int(g.integers(2**31)))
                if op == "layernorm":
                    x = spec.make_inputs(SHAPE, "cpu", torch.float32)[0]
                    gam = torch.full((x.shape[-1],), 2.0)
                    bet = torch.full((x.shape[-1],), 3.0)
                    a0 = ln_ref(x, gam, bet).float()
                    b = torch.nn.functional.layer_norm(
                        x.float(), (x.shape[-1],)) * 2.0 + 3.0
                    dev = (a0 * (1 + delta) - b).abs()
                    thr = 1e-4 + 1e-5 * b.abs()
                    caught = bool((dev > thr).any())
                else:
                    # cross_shape: 5 shapes drawn sequentially, exact
                    caught = False
                    for shape in spec.valid_shapes:
                        inputs = spec.make_inputs(shape, "cpu",
                                                  torch.float32)
                        f = (spec.run_reference(ref, inputs)
                             if isinstance(inputs, tuple) else ref(inputs))
                        dev = delta * f.abs()
                        thr = 1e-4 + 1e-4 * f.abs()
                        if bool((dev > thr).any()):
                            caught = True
                            break
                    # NOTE: candidate == (1+delta)*ref exactly, so
                    # deviation is delta|f| with no baseline residual.
                catches += caught
            obs = [r["caught"] for r in vbank
                   if r["op"] == op and r["mutant"] == vm]
            fracs[vm] = (catches / nrep, sum(obs), len(obs))
            print(f"  {op:14s} {vm}: predicted P = {catches/nrep:5.1%}   "
                  f"observed {sum(obs)}/{len(obs)}")
        vres[op] = fracs

    json.dump({"m_series": results,
               "v_series": {op: {k: list(v) for k, v in d.items()}
                            for op, d in vres.items()}},
              open(os.path.join(DATA, "response_law.json"), "w"), indent=1)
    print("\nwritten:", os.path.join(DATA, "response_law.json"))


if __name__ == "__main__":
    main()
