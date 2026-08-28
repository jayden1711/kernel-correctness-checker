"""
Design probe for the VERDICT-LEVEL near-miss family (v-series).

The m-series (near_miss_2026-08-28) straddles the adaptive-tolerance
boundary but every mutant is still verdict-caught by tighter checks. The
verdict boundary for a uniform (1+delta) scaling is

    delta*_verdict = min over every check in the pipeline of delta*_check

where delta*_check is the smallest delta at which that check fails. This
probe finds delta*_check by BISECTION THROUGH THE SHIPPED CHECK FUNCTIONS
themselves (property checks, cross_shape, weight_magnitude, the
perturbation base check and every spec adversarial variant, floors
included), run on CPU with fp32 emulations of the reference kernels in
the specs' own signatures. Deterministic seeds; 5 RNG draws per
stochastic check to get the draw spread (which is what limits how cleanly
the v-series can straddle).

Outputs data/design_verdict.json: per op the full delta* table, the
binding check, the second-binding check and their gap.

Run:  .venv/bin/python design_verdict_deltas.py
"""
import importlib
import json
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from verification.layer2_numeric_oracle.perturbation import (
    check_perturbation_tolerance)
from verification.layer2_numeric_oracle.shape_generalization import (
    check_weight_magnitude)
from verification.checker import _check_cross_shape

SHAPE = (64, 128)
OPS = ["layernorm", "softmax", "gelu", "l2norm", "sum_reduction"]


# ---- fp32 CPU emulations in the reference wrappers' signatures ----------

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


REFS = {"layernorm": ln_ref, "softmax": softmax_ref, "gelu": gelu_ref,
        "l2norm": l2norm_ref, "sum_reduction": sum_ref}


def scaled(fn, delta):
    def f(*args, **kw):
        return fn(*args, **kw) * (1.0 + delta)
    return f


def check_fns(op, spec, ref):
    """(name, callable(delta) -> passed) for every check the pipeline runs
    on this op that a uniform scaling can affect."""
    out = []
    inputs0 = None

    def base_inputs(seed):
        torch.manual_seed(seed)
        return spec.make_inputs(SHAPE, "cpu", torch.float32)

    # L2 property checks
    for pname, pfn in spec.algebraic_properties:
        def run(delta, pfn=pfn, s=0):
            torch.manual_seed(s)
            inputs = base_inputs(s)
            r = pfn(scaled(ref, delta), inputs)
            return bool(r[0])
        out.append((pname, run, False))          # deterministic-ish

    # cross_shape / weight_magnitude (shipped functions, spec-driven)
    def run_cs(delta, s=0):
        torch.manual_seed(s)
        r = _check_cross_shape(scaled(ref, delta), ref, spec)
        return bool(r[0])
    out.append(("cross_shape", run_cs, True))

    def run_wm(delta, s=0):
        torch.manual_seed(s)
        r = check_weight_magnitude(scaled(ref, delta), ref, spec)
        return bool(r[0])
    out.append(("weight_magnitude", run_wm, True))

    # perturbation base + adversarial variants (shipped function, floors in)
    def make_pert(adv=None, vname="perturbation_tolerance"):
        def run(delta, s=0):
            torch.manual_seed(s)
            inputs = base_inputs(s)
            if adv is not None:
                pairs = dict(spec.get_adversarial_inputs(inputs))
                ai = pairs[adv]
            else:
                ai = inputs
            if isinstance(ai, tuple):
                x, comps = ai[0], tuple(ai[1:])
                c = lambda t: spec.run_candidate(scaled(ref, delta),
                                                 (t,) + comps)
                r = lambda t: spec.run_reference(ref, (t,) + comps)
            else:
                x, comps = ai, ()
                c, r = scaled(ref, delta), ref
            ok, _ = check_perturbation_tolerance(c, r, x, op_name=op,
                                                 companions=comps)
            return bool(ok) if ok is not None else True
        return run
    out.append(("perturbation_tolerance", make_pert(), True))
    torch.manual_seed(0)
    for vname, _ in spec.get_adversarial_inputs(base_inputs(0)):
        out.append((f"adversarial_{vname}", make_pert(vname), True))
    return out


def bisect_delta(run, seed, lo=1e-8, hi=0.3):
    """Smallest delta with run(delta)==False (check fails); None if the
    check never fails in range (scaling-inert)."""
    def fails(d):
        return not run(d, s=seed)
    if not fails(hi):
        return None
    if fails(lo):
        return lo
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if fails(mid):
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def main():
    out = {}
    for op in OPS:
        spec = importlib.import_module(f"verification.specs.{op}").get_spec()
        ref = REFS[op]
        rows = []
        for name, run, stochastic in check_fns(op, spec, ref):
            seeds = range(5) if stochastic else range(1)
            ds = [bisect_delta(run, s) for s in seeds]
            ds = [d for d in ds if d is not None]
            if not ds:
                rows.append((name, None, None))
                continue
            med = sorted(ds)[len(ds) // 2]
            spread = max(ds) / min(ds) if min(ds) > 0 else float("inf")
            rows.append((name, med, spread))
        live = [(n, d, sp) for n, d, sp in rows if d is not None]
        live.sort(key=lambda t: t[1])
        out[op] = {"table": [(n, d, sp) for n, d, sp in rows],
                   "binding": live[0], "second": live[1] if len(live) > 1 else None}
        print(f"== {op}")
        for n, d, sp in rows:
            tag = ""
            if live and n == live[0][0]:
                tag = "   <-- BINDING"
            elif len(live) > 1 and n == live[1][0]:
                tag = "   <-- second"
            print(f"   {n:38s} delta* = "
                  f"{'inert' if d is None else f'{d:.3e}'}"
                  + (f"  (draw spread x{sp:.2f})" if d is not None and sp else "")
                  + tag)
        b, s2 = live[0], (live[1] if len(live) > 1 else None)
        gap = (s2[1] / b[1]) if s2 else float("inf")
        print(f"   verdict delta* = {b[1]:.3e} ({b[0]}); second at x{gap:.2f}"
              f"{' -- STRADDLE CONFOUNDED (gap < 2x)' if gap < 2 else ''}\n")
        out[op]["gap"] = gap
    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "design_verdict.json")
    json.dump(out, open(path, "w"), indent=1)
    print("written:", os.path.normpath(path))


if __name__ == "__main__":
    main()
