"""
Generates the VERDICT-LEVEL near-miss family:
TritonBench/near_miss/<op>/v{050,080,100,125,200}.py

Same mis-scaled-epilogue mechanism and kernel templates as the m-series
(imported from ../../near_miss_2026-08-28/probes/generate_mutants.py), but
DELTA is targeted at the op's BINDING check -- the smallest flip-delta in
the whole pipeline, found by bisection through the shipped check functions
(design_verdict_deltas.py):

    layernorm      affine_correctness             delta* = 1.937e-05
    softmax        adversarial_max_in_last_tile   delta* = 1.013e-06
                   (lockstep with adversarial_extreme_range, same floor)
    gelu           adversarial_near_global_min    delta* = 8.762e-06
    l2norm         cross_shape                    delta* = 3.412e-04
    sum_reduction  cross_shape                    delta* = 1.011e-04

Run:  .venv/bin/python generate_v_mutants.py     (idempotent)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "verification_runs",
                                "near_miss_2026-08-28", "probes"))
from generate_mutants import BODIES  # noqa: E402  (kernel templates)

DESIGN = os.path.join(HERE, "..", "data", "design_verdict.json")
TARGETS = {"v050": 0.5, "v080": 0.8, "v100": 1.0, "v125": 1.25, "v200": 2.0}

HEADER = '''"""
VERDICT-LEVEL near-miss mutant ({op}, target margin {margin}x of the
BINDING check `{binding}`, delta* = {dstar:.3e}).

The reference kernel with its output scaled by (1 + DELTA),
DELTA = {delta}. Unlike the m-series (which targets the adaptive
perturbation tolerance, 20-30x looser), DELTA here sits at {margin}x the
smallest flip-delta of ANY check in the pipeline, so the OVERALL VERDICT
straddles. Design: verification_runs/near_miss_verdict_2026-08-28/.
NOT part of the published corpus.
"""
import torch
import triton
import triton.language as tl

DELTA = {delta}

'''


def main():
    design = json.load(open(DESIGN))
    base = os.path.join(ROOT, "TritonBench", "near_miss")
    for op, body in BODIES.items():
        binding_name, dstar, _ = design[op]["binding"]
        opdir = os.path.join(base, op)
        for name, margin in TARGETS.items():
            delta = margin * dstar
            src = HEADER.format(op=op, margin=margin, binding=binding_name,
                                dstar=dstar, delta=repr(delta)) + body
            with open(os.path.join(opdir, f"{name}.py"), "w") as f:
                f.write(src)
            print(f"wrote near_miss/{op}/{name}.py  delta={delta:.3e} "
                  f"({binding_name})")


if __name__ == "__main__":
    main()
