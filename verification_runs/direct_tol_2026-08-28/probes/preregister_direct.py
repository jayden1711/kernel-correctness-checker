"""
PRE-REGISTERED predictions for the DIRECT-tolerance GPU run, computed and
banked BEFORE the arm executes (response-law machinery, CPU only).

Under KCC_STRUCTURAL_MODE=direct the perturbation tolerance is the
deterministic parent mean instead of a q95-of-20 draw, so the response-law
prediction for the m-series changes: the tol-draw randomness disappears and

    P_catch(m) = P_x( m * rho0 * M(x) > max(3 sigma(x) L(x) E(x), 1e-6) )

with only input-draw randomness left. The response-law round validated the
same functional WITH draw randomness against the banked probe-arm curves;
this file banks the DIRECT-arm predictions so the GPU run scores against a
committed number.

Also banks the predicted per-op tol ratio direct/probe-E (should be ~1.00)
for the corpus primaries.

Run:  .venv/bin/python preregister_direct.py
"""
import importlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verification_runs",
                                "response_law_2026-08-28", "probes"))
DATA = os.path.join(HERE, "..", "data")

from verification.layer2_numeric_oracle.structural_l import (
    row_norms, e_q95_direct)
from response_law import ensemble  # reuses REFS emulations
import response_law as rl

OPS = ["layernorm", "softmax", "gelu", "l2norm", "sum_reduction"]
MKEYS = ["m050", "m080", "m100", "m125", "m200"]
MARGINS = [0.5, 0.8, 1.0, 1.25, 2.0]
NREP = 3000


def ensemble_direct(op, nrep=NREP, seed=54321):
    spec = importlib.import_module(f"verification.specs.{op}").get_spec()
    ref = rl.REFS[op]
    g = np.random.default_rng(seed)
    Ms, tols = [], []
    for i in range(nrep):
        torch.manual_seed(int(g.integers(2**31)))
        inputs = spec.make_inputs((64, 128), "cpu", torch.float32)
        if isinstance(inputs, tuple):
            x, comps = inputs[0], list(inputs[1:])
            f = spec.run_reference(ref, inputs)
        else:
            x, comps = inputs, []
            f = ref(inputs)
        M = float(f.abs().max())
        x_std = float(x.float().std()) or 1.0
        sigma = 1e-3 * x_std
        rn = row_norms(op, x, comps)
        y = e_q95_direct(rn.double(), 20)
        tol = max(3.0 * sigma * float(rn.max()) * y, 1e-6)
        Ms.append(M)
        tols.append(tol)
    return np.array(Ms), np.array(tols)


def main():
    design = json.load(open(os.path.join(
        ROOT, "verification_runs", "near_miss_2026-08-28", "data",
        "design_deltas.json")))
    out = {}
    for op in OPS:
        Ms, tols = ensemble_direct(op)
        rho0 = design[op]["rho_median"]
        out[op] = {}
        for mk, m in zip(MKEYS, MARGINS):
            delta = design[op]["deltas"][str(m)]
            p = float(np.mean(delta * Ms > tols))
            out[op][mk] = p
        marg = rho0 * Ms / tols
        out[op]["margin_p5_p50_p95"] = list(np.percentile(marg, [5, 50, 95]))
        print(op, {k: (f"{v:.2f}" if isinstance(v, float) else v)
                   for k, v in out[op].items()})
    with open(os.path.join(DATA, "preregistered_direct_predictions.json"),
              "w") as f:
        json.dump(out, f, indent=1)
    print("banked BEFORE the GPU run.")


if __name__ == "__main__":
    main()
