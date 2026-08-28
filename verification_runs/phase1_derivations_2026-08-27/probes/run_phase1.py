"""Do the Phase-1 specs actually SEPARATE correct kernels from buggy ones?

Runs, per operator, the parts of the pipeline that are device-independent:
  L2  spec.algebraic_properties
  L3  spec.get_adversarial_inputs -> allclose against the reference
  L3  check_perturbation_tolerance on the base input

DELIBERATELY NOT RUN: Layer 1's AST checks. They test for Triton kernel
launches and would fail a torch candidate -- INCLUDING the correct one -- so
including them would manufacture a 100% catch rate that says nothing. Same
reason the plan flags them as an L2/L3 blocker.

Reported: catch rate on mutants, and false-positive rate on the CORRECT
implementation, which is the number that matters -- a battery that flags
everything is worthless.
"""
import importlib, json, os, sys, traceback
import torch

ROOT = "/Users/jaydenvasquez/Library/CloudStorage/GoogleDrive-jaydenvasquez1711@gmail.com/My Drive/kernel-correctness-checker"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutants.defs import OPS
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance

torch.manual_seed(0)
DEV, DT = "cpu", torch.float32
ATOL, RTOL = 1e-4, 1e-4


def judge(spec, cand, ref, inputs):
    """Return dict of check_name -> (passed, detail). passed False == flagged."""
    out = {}
    for pname, pfn in spec.algebraic_properties:
        try:
            r = pfn(cand, inputs)
            out[f"L2:{pname}"] = (bool(r[0]) if isinstance(r, (tuple, list)) else bool(r),
                                  r[1] if isinstance(r, (tuple, list)) and len(r) > 1 else "")
        except Exception as e:
            out[f"L2:{pname}"] = (False, f"{type(e).__name__}: {e}")
    try:
        for aname, ainp in spec.get_adversarial_inputs(inputs):
            try:
                c = spec.run_candidate(cand, ainp)
                r = spec.run_reference(ref, ainp)
                if c.shape != r.shape:
                    out[f"L3:adv_{aname}"] = (False, f"shape {tuple(c.shape)} vs {tuple(r.shape)}")
                    continue
                cf, rf = c.float(), r.float()
                fin = torch.isfinite(cf) & torch.isfinite(rf)
                if not torch.equal(torch.isfinite(cf), torch.isfinite(rf)):
                    out[f"L3:adv_{aname}"] = (False, "non-finite pattern differs")
                    continue
                ok = torch.allclose(cf[fin], rf[fin], atol=ATOL, rtol=RTOL) if fin.any() else True
                err = (cf[fin] - rf[fin]).abs().max().item() if fin.any() else 0.0
                out[f"L3:adv_{aname}"] = (ok, f"max_err={err:.3e}")
            except Exception as e:
                out[f"L3:adv_{aname}"] = (False, f"{type(e).__name__}: {e}")
    except Exception as e:
        out["L3:adv_setup"] = (False, f"{type(e).__name__}: {e}")

    try:
        prim = spec.primary_input(inputs)
        def _c(x): return spec.run_candidate(cand, ((x,) + inputs[1:]) if isinstance(inputs, tuple) else x)
        def _r(x): return spec.run_reference(ref, ((x,) + inputs[1:]) if isinstance(inputs, tuple) else x)
        comp = tuple(inputs[1:]) if isinstance(inputs, tuple) else ()
        res = check_perturbation_tolerance(_c, _r, prim, batch_samples=spec.batch_samples,
                                           op_name=spec.name, companions=comp)
        out["L3:perturbation"] = (bool(res[0]), str(res[1])[:90])
    except Exception as e:
        out["L3:perturbation"] = (False, f"{type(e).__name__}: {e}")
    return out


rows = []
print(f"{'operator':18s} {'FP(correct)':>12s} {'mutants caught':>15s}   first catching check")
print("-" * 100)
tot_m = tot_c = fp_ops = 0
for key, d in OPS.items():
    try:
        spec = importlib.import_module(f"verification.specs.{key}").get_spec()
        inputs = spec.make_inputs(spec.valid_shapes[0], DEV, DT)
    except Exception as e:
        print(f"{key:18s}  SPEC ERROR {type(e).__name__}: {e}"); continue

    cj = judge(spec, d["correct"], d["ref"], inputs)
    fp = [k for k, (p, _) in cj.items() if not p]
    if fp: fp_ops += 1

    caught, details = 0, []
    for mname, mfn in d["mutants"].items():
        mj = judge(spec, mfn, d["ref"], inputs)
        flagged = [k for k, (p, _) in mj.items() if not p and k not in fp]
        if flagged:
            caught += 1; details.append((mname, flagged[0], mj[flagged[0]][1]))
        else:
            details.append((mname, None, ""))
        rows.append(dict(op=key, mutant=mname, caught=bool(flagged),
                         flagged_by=flagged, fp_on_correct=fp))
    tot_m += len(d["mutants"]); tot_c += caught
    first = next((f"{m}->{c}" for m, c, _ in details if c), "-- NONE CAUGHT --")
    print(f"{key:18s} {len(fp):12d} {caught:>8d}/{len(d['mutants']):<6d}   {first[:52]}")
    if fp:
        print(f"{'':18s}   FP checks on CORRECT impl: {fp}")

print("-" * 100)
print(f"operators: {len(OPS)}   mutants: {tot_m}   caught: {tot_c} ({100*tot_c/tot_m:.1f}%)")
print(f"operators with a false positive on the CORRECT implementation: {fp_ops}")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_catch.jsonl"), "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
