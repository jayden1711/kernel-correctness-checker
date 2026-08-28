"""CPU predictions for the GPU out-of-sample run — computed and banked BEFORE
the GPU measurement is downloaded, from the same deterministic numpy inputs
attn_gram_gpu.py draws (default_rng([seed, N, D, op_idx])).

For each (op, shape, seed): exact float64 Jacobian of the reference math at
that Q/K/V, L = max row norm, and the Gram-law prediction
y_pred = E[q95_40(max |J z|)]/L with sd. y is delta_scale-free, so one
prediction covers all three delta_scale arms; the GPU y must match at every
delta_scale (scale-invariance falsification) and at every shape
(out-of-sample falsification).

Writes data/attn_gram_predictions.json.
"""

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from attn_gram_cpu import ref_fn, jacobian_rows, sim_exact, sim_m3  # noqa: E402

OPS = ["causal_flash_attention", "flash_attention",
       "scaled_dot_product_attention"]
OP_IDX = {op: i for i, op in enumerate(sorted(OPS))}
SHAPES = [(64, 32), (128, 64), (100, 32), (256, 16)]
SEEDS = [0, 1, 2]

torch.manual_seed(1)
out = []
for op in OPS:
    for (N, D) in SHAPES:
        for seed in SEEDS:
            rng = np.random.default_rng([seed, N, D, OP_IDX[op]])
            Qn = rng.normal(size=(N, D)).astype(np.float32)
            Kn = rng.normal(size=(N, D)).astype(np.float32)
            Vn = rng.normal(size=(N, D)).astype(np.float32)
            J = jacobian_rows(op, Qn, Kn, Vn)
            rn = J.norm(dim=1)
            L = rn.max().item()
            y_pred, y_sd = sim_exact(J.float(), L)
            m3_pred, _ = sim_m3(rn.float())
            out.append(dict(op=op, N=N, D=D, seed=seed, L_exact=L,
                            sigma_over_ds=float(torch.from_numpy(Qn).float().std()),
                            y_pred=y_pred, y_sd=y_sd, m3_pred=m3_pred))
            print(f"{op:30s} {N:4d}x{D:<3d} s{seed} L {L:.4f} "
                  f"y_pred {y_pred:.4f}+-{y_sd:.4f} m3/gram {m3_pred/y_pred:.3f}",
                  flush=True)

json.dump(out, open(os.path.join(HERE, "../data/attn_gram_predictions.json"), "w"),
          indent=1)
print("banked", len(out), "predictions")
