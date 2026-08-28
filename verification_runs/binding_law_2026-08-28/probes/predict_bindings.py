"""
The binding-check law: closed-form flip-deltas for every check in the
pipeline, with NO bisection.

CLAIM UNDER TEST. For a uniform (1+delta) output scaling, every check's
flip-delta has the closed form

    delta*_check = min_i  ( atol_c + rtol_c * |b_i| - |e0_i| ) / |v_i|

where (a, b) is the pair the comparator feeds torch.allclose, e0 = a - b is
the baseline residual at delta = 0 (fp noise; exactly 0 whenever both sides
come from the same scaled function), and v_i = d(a_i - b_i)/d(delta) is the
DERIVED deviation velocity:

    v_i = k * s_i   with k the scaling degree of the compared statistic
          (k = 1 for the output itself, sums, norms, RMS; k = 2 for
           variance; k = 0 -- check inert -- whenever both sides are
           computed from the same scaled candidate, which covers every
           *_invariance / equivariance / monotonicity / gamma_correctness
           check by construction).

For the perturbation family the threshold is not a constant but the
adaptive tolerance, and the law composes with the structural parent
(the same object H1/Gram derived):

    delta*_pert = max( scale * sigma * L * E[q95_n](profile), 1e-6 ) / max|f_v|

with E[q95_n] the deterministic order-statistic integral of the parent CDF
F(t) = prod_i (2 Phi(t/w_i) - 1)  (e_q95_direct, validated against the
banked native bank at R^2(log) = 0.997 in ../direct_tol_2026-08-28).

precision_coercion is the one two-arm comparator: fail iff
err32(delta) > atol AND err32(delta) >= 0.9 * err16(delta), with
errP(delta) = max_i |e0P_i + delta * fP_i| an exact elementwise function of
two forward evaluations; delta* is its smallest root, found on a delta grid
of the DERIVED model (no check bisection).

STAGES
  validate : predict the full table for the five v-series ops and score
             against the banked bisection ground truth
             (../near_miss_verdict_2026-08-28/data/design_verdict.json).
  blind    : predict rmsnorm + frobenius_norm tables FIRST (written to
             data/blind_predictions.json), then run the design probe's own
             bisection machinery on those two ops as ground truth
             (data/blind_truth.json) and score.

Run:  .venv/bin/python predict_bindings.py
"""
import importlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(HERE, "..", "data")

from verification.layer2_numeric_oracle.structural_l import row_norms
from verification.layer2_numeric_oracle.shape_generalization import (
    _make_weight_variants)

# the validated deterministic parent integral
sys.path.insert(0, os.path.join(ROOT, "verification_runs",
                                "direct_tol_2026-08-28", "probes"))
from direct_e import e_q95_direct

SHAPE = (64, 128)
HI = 0.3          # the design probe's bisection ceiling: predictions above
                  # HI are "inert in range"

# ---- fp32 CPU emulations, IDENTICAL to the design probe's ----------------

def ln_ref(x, gamma, beta, eps=1e-5):
    m = x.mean(-1, keepdim=True)
    v = ((x - m) ** 2).mean(-1, keepdim=True)
    return (x - m) / torch.sqrt(v + eps) * gamma + beta


def softmax_ref(x):
    m = x.max(-1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(-1, keepdim=True)


def gelu_ref(x):
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def l2norm_ref(x, eps=1e-12):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def sum_ref(x):
    return x.sum(-1)


# blind-test emulations (transcribed from TritonBench/reference/*.py)

def rmsnorm_ref(x, gamma, eps=1e-5):
    rms = torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
    return x / rms * gamma


def frob_ref(x, eps=1e-12):
    return x / (torch.sqrt((x * x).sum()) + eps)


REFS = {"layernorm": ln_ref, "softmax": softmax_ref, "gelu": gelu_ref,
        "l2norm": l2norm_ref, "sum_reduction": sum_ref,
        "rmsnorm": rmsnorm_ref, "frobenius_norm": frob_ref}

VALIDATE_OPS = ["layernorm", "softmax", "gelu", "l2norm", "sum_reduction"]
BLIND_OPS = ["rmsnorm", "frobenius_norm"]


# ---- the closed-form solvers ---------------------------------------------

def k1_delta(stat, target, atol, rtol):
    """delta* for |s(1+delta) - b| > atol + rtol|b|, elementwise min.
    Exact for a degree-1 statistic. Baseline residual folded in."""
    s = stat.double().flatten()
    b = (target.double().flatten() if torch.is_tensor(target)
         else torch.full_like(s, float(target)))
    thr = atol + rtol * b.abs()
    e0 = s - b
    # deviation(delta) = |e0 + delta*s|; worst direction is sign(s)
    # aligned with e0 -> crossing at (thr - |e0|)/|s| when e0 aligned,
    # (thr + |e0|)/|s| against; the check fails when ANY element crosses,
    # so take the aligned (smaller) root, floored at 0.
    room = (thr - e0.abs()).clamp_min(0.0)
    with np.errstate(divide="ignore"):
        d = np.where(s.abs().numpy() > 0,
                     room.numpy() / s.abs().numpy(), np.inf)
    return float(np.min(d)) if d.size else float("inf")


def k2_delta(stat, target, atol, rtol):
    """delta* for |s(1+delta)^2 - b| > atol + rtol|b| (variance-type)."""
    s = stat.double().flatten()
    b = (target.double().flatten() if torch.is_tensor(target)
         else torch.full_like(s, float(target)))
    thr = atol + rtol * b.abs()
    # (1+d)^2 = (b + thr)/s  (growth direction; s>0 for variances)
    lim = ((b + thr) / s).clamp_min(0.0)
    d = torch.sqrt(lim) - 1.0
    d = torch.where(s > 0, d, torch.full_like(d, float("inf")))
    return float(d.clamp_min(0.0).min()) if d.numel() else float("inf")


def affine_delta(a0, b, atol, rtol):
    """delta* for |a0(1+delta) - b| > atol + rtol|b|, a0 ~ b + e0."""
    a0 = a0.double().flatten()
    b = b.double().flatten()
    thr = atol + rtol * b.abs()
    e0 = (a0 - b).abs()
    room = (thr - e0).clamp_min(0.0)
    d = torch.where(a0.abs() > 0, room / a0.abs(),
                    torch.full_like(a0, float("inf")))
    return float(d.min())


def precision_delta(f32, ref, f16, atol=1e-3, factor=0.9):
    """Smallest delta with err32(d) > atol AND err32(d) >= factor*err16(d),
    errP(d) = max_i |e0P_i + d * fP_i|. Exact model, grid-solved."""
    e32 = (f32.double() - ref.double()).flatten()
    v32 = f32.double().flatten()
    have16 = f16 is not None
    if have16:
        e16 = (f16.double() - ref.double()).flatten()
        v16 = f16.double().flatten()
    grid = np.logspace(-8, math.log10(HI), 4000)
    for d in grid:
        err32 = float((e32 + d * v32).abs().max())
        if err32 <= atol:
            continue
        if not have16:
            return float(d)
        err16 = float((e16 + d * v16).abs().max())
        if err32 >= factor * err16:
            return float(d)
    return None


def scaled_dev_delta(f, atol, rtol):
    """delta* for the candidate == (1+delta)*ref comparators
    (cross_shape per shape, weight_magnitude per variant): deviation is
    EXACTLY delta*|f_i|; fail iff delta|f_i| > atol + rtol|f_i|."""
    fa = f.double().abs().flatten()
    fa = fa[fa > 0]
    if fa.numel() == 0:
        return float("inf")
    return float((atol / fa + rtol).min())


# ---- per-op table prediction ---------------------------------------------

INERT = {"scale_invariance", "shift_invariance", "monotonicity",
         "permutation_invariance", "scale_linearity",
         "positive_scale_invariance", "zero_at_origin",
         "monotonic_nonneg", "gamma_correctness"}


def predict_table(op):
    spec = importlib.import_module(f"verification.specs.{op}").get_spec()
    ref = REFS[op]
    rows = []

    def base_inputs(seed):
        torch.manual_seed(seed)
        return spec.make_inputs(SHAPE, "cpu", torch.float32)

    # ---- property checks (seed 0, matching the design probe) -------------
    for pname, _ in spec.algebraic_properties:
        torch.manual_seed(0)
        inputs = base_inputs(0)
        d = None
        note = ""
        if pname in INERT:
            d = None
            note = "derived inert (comparator is scaled-vs-scaled or zero)"
        elif op == "layernorm" and pname == "zero_mean":
            x = inputs[0]
            out = ln_ref(x, torch.ones(x.shape[-1]), torch.zeros(x.shape[-1]))
            d = k1_delta(out.mean(-1), 0.0, 1e-4, 1e-5)
        elif op == "layernorm" and pname == "unit_variance":
            x = inputs[0]
            out = ln_ref(x, torch.ones(x.shape[-1]), torch.zeros(x.shape[-1]))
            d = k2_delta(out.float().var(-1, unbiased=False), 1.0, 1e-3, 1e-5)
        elif op == "layernorm" and pname == "affine_correctness":
            x = inputs[0]
            g = torch.full((x.shape[-1],), 2.0)
            b = torch.full((x.shape[-1],), 3.0)
            a0 = ln_ref(x, g, b).float()
            expected = torch.nn.functional.layer_norm(
                x.float(), (x.shape[-1],)) * 2.0 + 3.0
            d = affine_delta(a0, expected, 1e-4, 1e-5)
        elif op == "layernorm" and pname == "precision_coercion":
            x = inputs[0]
            g1 = torch.ones(x.shape[-1]); b0 = torch.zeros(x.shape[-1])
            refq = torch.nn.functional.layer_norm(
                x.double(), (x.shape[-1],)).float()
            f32 = ln_ref(x.float(), g1, b0).float()
            d = precision_delta(f32, refq, None)     # layernorm: no fp16 arm
        elif op == "softmax" and pname == "rows_sum_to_one":
            x = inputs
            out = softmax_ref(x)
            d = k1_delta(out.sum(-1), torch.ones(x.shape[0]), 1e-4, 1e-5)
        elif op == "softmax" and pname == "precision_coercion":
            x = inputs
            refq = softmax_ref(x.double()).float()
            f32 = softmax_ref(x.float()).float()
            try:
                f16 = softmax_ref(x.half()).float()
            except Exception:
                f16 = None
            d = precision_delta(f32, refq, f16)
        elif op == "l2norm" and pname == "unit_l2_norm":
            x = inputs
            out = l2norm_ref(x)
            d = k1_delta(out.norm(dim=-1), torch.ones(x.shape[0]), 1e-3, 1e-5)
        elif op == "rmsnorm" and pname == "unit_rms":
            x = inputs[0]
            out = rmsnorm_ref(x, torch.ones(x.shape[-1]))
            rms = out.float().pow(2).mean(-1).sqrt()
            d = k1_delta(rms, torch.ones(x.shape[0]), 1e-3, 1e-5)
        elif op == "rmsnorm" and pname == "precision_coercion":
            x = inputs[0]
            x_d = x.double()
            refq = (x_d / (x_d.pow(2).mean(-1, keepdim=True).sqrt()
                           + 1e-5)).float()
            g1 = torch.ones(x.shape[-1])
            f32 = rmsnorm_ref(x.float(), g1).float()
            try:
                f16 = rmsnorm_ref(x.half(), g1.half()).float()
            except Exception:
                f16 = None
            d = precision_delta(f32, refq, f16)
        elif op == "frobenius_norm" and pname == "unit_frobenius_norm":
            x = inputs
            out = frob_ref(x)
            n = out.float().norm().reshape(1)
            d = k1_delta(n, 1.0, 1e-3, 0.0)   # strict |.-1| < atol, no rtol
        else:
            note = "NO RULE -- unpredicted"
        if d is not None and d > HI:
            d, note = None, note + " (crossing above bisection range)"
        rows.append((pname, d, note))

    # ---- cross_shape (5 seeds, median) -----------------------------------
    ds = []
    for s in range(5):
        torch.manual_seed(s)
        per_shape = []
        for shape in spec.valid_shapes:
            inputs = spec.make_inputs(shape, "cpu", torch.float32)
            f = spec.run_reference(ref, inputs)
            per_shape.append(scaled_dev_delta(f, 1e-4, 1e-4))
        ds.append(min(per_shape))
    rows.append(("cross_shape", float(np.median(ds)), ""))

    # ---- weight_magnitude (5 seeds, median) ------------------------------
    ds = []
    for s in range(5):
        torch.manual_seed(s)
        base = spec.make_inputs(spec.valid_shapes[0], "cpu", torch.float32)
        primary = spec.primary_input(base)
        variants = _make_weight_variants(primary)
        per_v = []
        for vname, adv in variants.items():
            adv_inputs = ((adv,) + base[1:]) if isinstance(base, tuple) else adv
            try:
                f = spec.run_reference(ref, adv_inputs)
            except Exception:
                continue
            per_v.append(scaled_dev_delta(f, 1e-3, 1e-3))
        if per_v:
            ds.append(min(per_v))
    rows.append(("weight_magnitude", float(np.median(ds)) if ds else None, ""))

    # ---- perturbation family (parent-composed; 5 seeds, median) ----------
    def pert_pred(seed, adv_name=None):
        torch.manual_seed(seed)
        inputs = base_inputs(seed)
        if adv_name is not None:
            pairs = dict(spec.get_adversarial_inputs(inputs))
            ai = pairs[adv_name]
        else:
            ai = inputs
        if isinstance(ai, tuple):
            x, comps = ai[0], list(ai[1:])
            f = spec.run_reference(ref, tuple([x] + comps))
        else:
            x, comps = ai, []
            f = ref(x)
        M = float(f.abs().max())
        if M == 0:
            return None, 0.0
        x_std = float(x.float().std())
        if x_std == 0:
            x_std = 1.0
        sigma = 1e-3 * x_std
        try:
            rn = row_norms(op, x, comps)
        except Exception:
            return None, M
        if rn is None or rn.numel() == 0 or not torch.isfinite(rn).all():
            return None, M
        rn = rn.double().numpy().ravel()
        rn = rn[rn > 0]
        if rn.size == 0:
            tol = 1e-6
        else:
            y = e_q95_direct(rn, 20)
            tol = max(3.0 * sigma * float(np.max(rn)) * y, 1e-6)
        return tol / M, M

    names = [("perturbation_tolerance", None)]
    torch.manual_seed(0)
    for vname, _ in spec.get_adversarial_inputs(base_inputs(0)):
        names.append((f"adversarial_{vname}", vname))
    for label, adv in names:
        preds = []
        floored = False
        for s in range(5):
            p, M = pert_pred(s, adv)
            if p is not None:
                preds.append(p)
                if abs(p * M - 1e-6) < 1e-12:
                    floored = True
        if preds:
            d = float(np.median(preds))
            note = "floor-bound" if floored else ""
            if d > HI:
                rows.append((label, None, note + " (above range)"))
            else:
                rows.append((label, d, note))
        else:
            rows.append((label, None, "parent declined"))

    live = sorted([(n, d) for n, d, _ in rows if d is not None],
                  key=lambda t: t[1])
    return {"table": [(n, d, note) for n, d, note in rows],
            "binding": live[0] if live else None,
            "second": live[1] if len(live) > 1 else None,
            "gap": (live[1][1] / live[0][1]) if len(live) > 1 else None}


# ---- scoring against ground truth ----------------------------------------

def score(pred, truth, opname):
    tmap = {n: (d, sp) for n, d, sp in truth["table"]}
    print(f"== {opname}")
    ratios = []
    agree_inert = 0
    n_inert = 0
    mismatches = []
    for n, d, note in pred["table"]:
        if n not in tmap:
            print(f"   {n:40s} (not in ground truth)")
            continue
        td, spread = tmap[n]
        if td is None and d is None:
            agree_inert += 1; n_inert += 1
            print(f"   {n:40s} inert / inert  OK   {note}")
        elif td is None or d is None:
            n_inert += td is None
            mismatches.append(n)
            print(f"   {n:40s} pred="
                  f"{'inert' if d is None else f'{d:.3e}'} vs "
                  f"truth={'inert' if td is None else f'{td:.3e}'}  "
                  f"MISMATCH  {note}")
        else:
            r = d / td
            ratios.append((n, r, spread))
            flag = "OK " if 1 / 1.5 <= r <= 1.5 else "OFF"
            print(f"   {n:40s} pred={d:.3e} truth={td:.3e} "
                  f"ratio={r:5.2f} (draw spread x{spread:.2f}) {flag} {note}")
    pb = pred["binding"][0] if pred["binding"] else None
    tb = truth["binding"][0]
    # softmax lockstep: either of the tied pair counts
    tied = [n for n, d, _ in truth["table"]
            if d is not None and abs(d / truth["binding"][1] - 1) < 0.02]
    ok = pb in tied
    print(f"   BINDING pred={pb} truth={tb} "
          f"{'MATCH' if ok else 'MISS'} (tied set: {tied})")
    return ratios, ok, mismatches


def main():
    os.makedirs(DATA, exist_ok=True)
    torch.set_num_threads(4)

    # ---------------- stage 1: validate on the five banked ops ------------
    banked = json.load(open(os.path.join(
        ROOT, "verification_runs", "near_miss_verdict_2026-08-28",
        "data", "design_verdict.json")))
    all_ratios, bind_ok = [], 0
    preds = {}
    for op in VALIDATE_OPS:
        p = predict_table(op)
        preds[op] = p
        r, ok, mm = score(p, banked[op], op)
        all_ratios += [x[1] for x in r]
        bind_ok += ok
    print(f"\nVALIDATE: binding match {bind_ok}/{len(VALIDATE_OPS)}; "
          f"finite-entry ratio p5/p50/p95 = "
          f"{np.percentile(all_ratios,5):.2f}/"
          f"{np.percentile(all_ratios,50):.2f}/"
          f"{np.percentile(all_ratios,95):.2f}  (n={len(all_ratios)})")
    json.dump(preds, open(os.path.join(DATA, "validate_predictions.json"),
                          "w"), indent=1)

    # ---------------- stage 2: blind -- predict FIRST ---------------------
    blind = {}
    for op in BLIND_OPS:
        blind[op] = predict_table(op)
    json.dump(blind, open(os.path.join(DATA, "blind_predictions.json"), "w"),
              indent=1)
    print("\nblind predictions written BEFORE ground truth:")
    for op in BLIND_OPS:
        print(f"  {op}: binding pred = {blind[op]['binding']}")
        for n, d, note in blind[op]["table"]:
            print(f"     {n:40s} "
                  f"{'inert' if d is None else format(d, '.3e')}  {note}")

    # ---------------- stage 2b: ground truth by bisection -----------------
    dp_path = os.path.join(ROOT, "verification_runs",
                           "near_miss_verdict_2026-08-28", "probes",
                           "design_verdict_deltas.py")
    sp = importlib.util.spec_from_file_location("design_probe", dp_path)
    dp = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(dp)
    dp.REFS["rmsnorm"] = rmsnorm_ref
    dp.REFS["frobenius_norm"] = frob_ref

    truth = {}
    for op in BLIND_OPS:
        spec = importlib.import_module(f"verification.specs.{op}").get_spec()
        ref = dp.REFS[op]
        rows = []
        for name, run, stochastic in dp.check_fns(op, spec, ref):
            seeds = range(5) if stochastic else range(1)
            ds = [dp.bisect_delta(run, s) for s in seeds]
            ds = [d for d in ds if d is not None]
            if not ds:
                rows.append((name, None, None))
                continue
            med = sorted(ds)[len(ds) // 2]
            spread = max(ds) / min(ds) if min(ds) > 0 else float("inf")
            rows.append((name, med, spread))
        live = [(n, d, sp_) for n, d, sp_ in rows if d is not None]
        live.sort(key=lambda t: t[1])
        truth[op] = {"table": rows, "binding": live[0],
                     "second": live[1] if len(live) > 1 else None}
    json.dump(truth, open(os.path.join(DATA, "blind_truth.json"), "w"),
              indent=1)

    print("\n---- BLIND SCORING ----")
    br, bok = [], 0
    for op in BLIND_OPS:
        r, ok, mm = score(blind[op], truth[op], op)
        br += [x[1] for x in r]
        bok += ok
    print(f"\nBLIND: binding match {bok}/{len(BLIND_OPS)}; "
          f"ratio p50 = {np.percentile(br,50):.2f} (n={len(br)})")


if __name__ == "__main__":
    main()
