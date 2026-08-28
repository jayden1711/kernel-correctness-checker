"""
GPU validation of the near-miss mutant family (T4, PYTHONPATH=/content).

Per (op, target margin) mutant, 10 seeds at the corpus shape (64,128):
  - runs the SHIPPED check_perturbation_tolerance (scale 3.0, P95,
    delta_scale 1e-3) candidate-vs-reference and records max_err,
    adaptive_tol, the realized margin, and the verdict;
  - runs the full KernelChecker battery at 3 seeds and records which
    checks fail (the near-miss scaling may trip property checks too --
    that is measured, not assumed).

Writes /content/nm/near_miss_gpu.json.
"""
import importlib
import json
import os
import re
import sys

sys.path.insert(0, "/content")

import torch

from verification.layer2_numeric_oracle.perturbation import (
    check_perturbation_tolerance)
from verification.checker import KernelChecker

assert torch.cuda.is_available()

OPS = ["layernorm", "softmax", "gelu", "l2norm", "sum_reduction"]
MARGINS = ["m050", "m080", "m100", "m125", "m200"]
SHAPE = (64, 128)
RE_ERR = re.compile(r"max_err=(\d+\.\d+)")
RE_TOL = re.compile(r"adaptive_tol=(\d+\.\d+)")

out = {"records": []}

for op in OPS:
    spec = importlib.import_module(f"verification.specs.{op}").get_spec()
    ref_mod = importlib.import_module(f"TritonBench.reference.{op}")
    ref_fn = getattr(ref_mod, op)
    for mname in MARGINS:
        mod = importlib.import_module(f"TritonBench.near_miss.{op}.{mname}")
        cand_fn = getattr(mod, op)
        raw_kernel = next(v for k, v in vars(mod).items()
                          if k != op and "kernel" in k.lower())
        for seed in range(10):
            torch.manual_seed(seed)
            inputs = spec.make_inputs(SHAPE, "cuda", torch.float32)
            if isinstance(inputs, tuple):
                x, comps = inputs[0], tuple(inputs[1:])
                cfn = lambda t: spec.run_candidate(cand_fn, (t,) + comps)
                rfn = lambda t: spec.run_reference(ref_fn, (t,) + comps)
            else:
                x, comps = inputs, ()
                cfn, rfn = cand_fn, ref_fn
            direct_err = float((cfn(x) - rfn(x)).abs().max())
            passed, detail = check_perturbation_tolerance(
                cfn, rfn, x, op_name=op, companions=comps)
            me, mt = RE_ERR.search(detail or ""), RE_TOL.search(detail or "")
            rec = {"op": op, "mutant": mname, "seed": seed,
                   "direct_err": direct_err,
                   "max_err": float(me.group(1)) if me else None,
                   "tol": float(mt.group(1)) if mt else None,
                   "pert_passed": bool(passed)}
            if rec["tol"]:
                rec["margin"] = direct_err / rec["tol"]
            out["records"].append(rec)
            if seed < 3:
                torch.manual_seed(1000 + seed)
                inputs2 = spec.make_inputs(SHAPE, "cuda", torch.float32)
                results = KernelChecker(spec).run(cand_fn, raw_kernel,
                                                  ref_fn, inputs2)
                out["records"].append(
                    {"op": op, "mutant": mname, "seed": 1000 + seed,
                     "full_battery": True,
                     "caught": (not all(r.passed for r in results)),
                     "failed": [f"[L{r.layer}]{r.check_name}"
                                for r in results if not r.passed]})
        print(f"{op}/{mname} done", flush=True)

os.makedirs("/content/nm", exist_ok=True)
json.dump(out, open("/content/nm/near_miss_gpu.json", "w"))
print("NEAR-MISS-GPU-OK")
