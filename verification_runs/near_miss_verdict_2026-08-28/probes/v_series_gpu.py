"""
GPU validation of the VERDICT-LEVEL near-miss family (T4,
PYTHONPATH=/content).

Per (op, vNNN) mutant, 10 seeds at (64,128):
  - FULL KernelChecker battery -> the verdict (caught / missed) and the
    set of failing checks. The verdict catch-rate curve across the design
    ladder is the deliverable: does the family straddle the VERDICT
    boundary?
  - realized margin of the DESIGN-BINDING check, computed directly from
    the mutant's delta and the same quantities the check compares
    (allclose thresholds / perturbation tolerances measured in place).

Writes /content/nmv/v_series_gpu.json.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, "/content")

import torch

from verification.checker import KernelChecker
from verification.layer2_numeric_oracle.perturbation import (
    check_perturbation_tolerance)

assert torch.cuda.is_available()

OPS = ["layernorm", "softmax", "gelu", "l2norm", "sum_reduction"]
MARGINS = ["v050", "v080", "v100", "v125", "v200"]
SHAPE = (64, 128)


def realized_margin(op, spec, ref_fn, delta, seed):
    """delta-scaling margin of the op's design-binding check, measured on
    the same GPU draws the check would use."""
    torch.manual_seed(seed)
    inputs = spec.make_inputs(SHAPE, "cuda", torch.float32)
    if op == "layernorm":
        # affine_correctness: out=(1+d)*(2*norm+3) vs expected, allclose
        # atol=1e-4 rtol=1e-5
        x = inputs[0]
        norm = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],))
        v = (norm * 2.0 + 3.0).abs()
        return float((delta * v / (1e-4 + 1e-5 * v)).max())
    if op in ("l2norm", "sum_reduction"):
        # cross_shape: allclose atol=1e-4 rtol=1e-4 over valid_shapes
        worst = 0.0
        for shp in spec.valid_shapes:
            ins = spec.make_inputs(shp, "cuda", torch.float32)
            y = spec.run_reference(ref_fn, ins).abs()
            worst = max(worst, float((delta * y / (1e-4 + 1e-4 * y)).max()))
        return worst
    # softmax / gelu: floor-bound adversarial variant; margin =
    # delta*max|f(x_v)| / tol_v with tol_v measured by the shipped probe
    vname = ("max_in_last_tile" if op == "softmax" else "near_global_min")
    pairs = dict(spec.get_adversarial_inputs(inputs))
    ai = pairs[vname]
    if isinstance(ai, tuple):
        x, comps = ai[0], tuple(ai[1:])
        r = lambda t: spec.run_reference(ref_fn, (t,) + comps)
    else:
        x, comps = ai, ()
        r = ref_fn
    _, detail = check_perturbation_tolerance(r, r, x, op_name=op,
                                             companions=comps)
    import re
    m = re.search(r"adaptive_tol=(\d+\.\d+)", detail or "")
    tol = float(m.group(1)) if m else 1e-6
    tol = max(tol, 1e-6)
    M = float(r(x).abs().max())
    return delta * M / tol


out = {"records": []}
for op in OPS:
    spec = importlib.import_module(f"verification.specs.{op}").get_spec()
    ref_mod = importlib.import_module(f"TritonBench.reference.{op}")
    ref_fn = getattr(ref_mod, op)
    for mname in MARGINS:
        mod = importlib.import_module(f"TritonBench.near_miss.{op}.{mname}")
        cand_fn = getattr(mod, op)
        delta = mod.DELTA
        raw_kernel = next(v for k, v in vars(mod).items()
                          if k != op and "kernel" in k.lower())
        for seed in range(10):
            torch.manual_seed(seed)
            inputs = spec.make_inputs(SHAPE, "cuda", torch.float32)
            results = KernelChecker(spec).run(cand_fn, raw_kernel, ref_fn,
                                              inputs)
            failed = [f"[L{r.layer}]{r.check_name}" for r in results
                      if not r.passed]
            rm = realized_margin(op, spec, ref_fn, delta, 500 + seed)
            out["records"].append({"op": op, "mutant": mname, "seed": seed,
                                   "delta": delta, "caught": bool(failed),
                                   "failed": failed,
                                   "realized_margin": rm})
        print(f"{op}/{mname} done", flush=True)

os.makedirs("/content/nmv", exist_ok=True)
json.dump(out, open("/content/nmv/v_series_gpu.json", "w"))
print("V-SERIES-GPU-OK")
