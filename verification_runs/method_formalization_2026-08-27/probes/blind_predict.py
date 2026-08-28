"""
BLIND GENERALIZATION TEST, stage 1 (CPU): pre-registered predictions for
`logcumsumexp` -- an operator in NO prior round, NO spec, NO banked
measurement anywhere in this project (verified by grep before writing this).

The operator-agnostic procedure under test (METHOD.md steps a-d), applied
sight-unseen:

  (a) STRUCTURAL L. f(x)[r, i] = log sum_{j<=i} exp(x[r, j]) per row.
      dfi/dxj = exp(x_j - f_i) for j <= i, else 0 -- each Jacobian row is a
      softmax over a PREFIX, so rows are nonnegative with sum 1, and
      L = max_i ||J_i||_2 in (C^{-1/2}, 1]. Input-dependent, like attention;
      mechanically derivable, like everything built from linear ops + smooth
      nonlinearities with known Jacobians.
  (b) GRAM STRUCTURE. (J J^T)_{ik} = <p^(i), p^(k)> with p^(i) the prefix
      softmax: NESTED overlapping supports -- the scan family's Brownian
      nesting, but softmax-weighted and input-dependent. Classification
      predicted from structure alone: CORRELATED-ROW family; the
      independence-assuming M3 baseline must OVER-predict y (positive
      correlations reduce the effective number of independent maxima), by a
      factor between 1 and the exactly-linear scans' 1.231.
  (c) The full-distribution prediction y_pred = E[q95_40(max |J z|)] / L
      needs no closed form: rows of x are independent, J is block-diagonal
      per row, and the exact per-row Jacobian is the closed form
      J^(r) = tril(exp(x_j - f_i)) -- materialized here in float64 and
      SAMPLED, zero fitted constants. Paired-side prediction: the 40 deltas
      the GPU stage will use are drawn HERE (numpy, banked) and their exact
      directional derivatives s_lin = ||J d||_inf are banked with them.
  (d) Validation happens against torch.logcumsumexp's shipped CUDA kernel
      (an implementation this project did not write), stage 2.

Predictions are written BEFORE the GPU stage runs and are not edited after
-- the compare stage reads both files and reports z-scores.

Configs: 3 shapes x 3 input regimes x 2 seeds = 18 invocations.
Regimes: `randn` (primary), `large_magnitude` (x*50 -- prefix softmax nearly
one-hot: the analogue of the attention saturation input; the METHOD predicts
the paired ratio stays ~1 here because a one-hot-weighted prefix LSE is
locally a smooth selection, NOT the attention failure mode -- registering
that prediction is part of the blind test), `ascending` (sorted rows: the
running max dominates, rows near one-hot deterministically).

Outputs:
  data/blind_inputs.npz          x (float32) and the 40 deltas per config
  data/blind_predictions.json    y_pred mean/sd, L, y_M3, s_lin per delta
"""

import json
import math
import zlib
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NS = 40
NREP = 400
DELTA_SCALE = 1e-3
Q = 0.95

SHAPES = [(64, 256), (32, 333), (16, 512)]
REGIMES = ["randn", "large_magnitude", "ascending"]
SEEDS = [0, 1]


def make_input(rng, shape, regime):
    x = rng.standard_normal(shape).astype(np.float32)
    if regime == "large_magnitude":
        x = x * 50.0
    elif regime == "ascending":
        x = np.sort(x, axis=1)
    return x


def exact_row_jacobians(x64):
    """List of per-row (C, C) float64 Jacobians: J = tril(exp(x_j - f_i))."""
    f = torch.logcumsumexp(x64, dim=1)
    out = []
    for r in range(x64.shape[0]):
        J = torch.exp(x64[r].unsqueeze(0) - f[r].unsqueeze(1))
        out.append(torch.tril(J))
    return out


def q95_np(v):
    """torch-convention linear-interpolation quantile, replicated in numpy."""
    v = np.sort(v, axis=-1)
    h = Q * (v.shape[-1] - 1)
    lo = int(math.floor(h))
    fr = h - lo
    return v[..., lo] * (1 - fr) + v[..., lo + 1] * fr


def main():
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(20260827)
    bank = {}
    preds = []
    for shape in SHAPES:
        for regime in REGIMES:
            for seed in SEEDS:
                key = f"{shape[0]}x{shape[1]}_{regime}_s{seed}"
                # zlib.crc32, not hash(): str hashing is per-process
                # randomised and would silently unpin the pre-registration.
                crng = np.random.default_rng(
                    zlib.crc32(key.encode()) % (2**31))
                x = make_input(crng, shape, regime)
                sigma = float(x.std())
                deltas = (crng.standard_normal((NS,) + shape)
                          .astype(np.float32) * DELTA_SCALE * sigma)
                bank[key + "_x"] = x
                bank[key + "_d"] = deltas

                x64 = torch.from_numpy(x).double()
                Js = exact_row_jacobians(x64)
                L = max(float(J.pow(2).sum(dim=1).max().sqrt()) for J in Js)

                # Paired side: exact s_lin per banked delta.
                s_lin = []
                for k in range(NS):
                    d = torch.from_numpy(deltas[k]).double()
                    m = 0.0
                    for r, J in enumerate(Js):
                        m = max(m, float((J @ d[r]).abs().max()))
                    s_lin.append(m)

                # Distributional side: y = E[q95_40(max |J z|)] / L over unit
                # Gaussian z (sigma scales out of y by construction).
                R, C = shape
                need = NREP * NS
                s_all = np.empty(need, dtype=np.float64)
                Jnp = [J.numpy() for J in Js]
                chunk = 2000
                done = 0
                while done < need:
                    b = min(chunk, need - done)
                    z = crng.standard_normal((R, b, C))
                    m = np.zeros(b)
                    for r in range(R):
                        m = np.maximum(m, np.abs(z[r] @ Jnp[r].T).max(axis=1))
                    s_all[done:done + b] = m
                    done += b
                y = q95_np(s_all.reshape(NREP, NS)) / L
                y_mean, y_sd = float(y.mean()), float(y.std())

                # M3 baseline: independent-max over the row-norm profile.
                norms = np.concatenate(
                    [np.sqrt((J.numpy() ** 2).sum(axis=1)) for J in Js])
                g = crng.standard_normal((NREP, NS, norms.size))
                s_m3 = np.abs(g * norms).max(axis=2)
                y_m3 = float((q95_np(s_m3) / L).mean())

                preds.append({
                    "key": key, "shape": list(shape), "regime": regime,
                    "seed": seed, "sigma": sigma, "L": L,
                    "y_pred_mean": y_mean, "y_pred_sd": y_sd,
                    "y_m3_mean": y_m3, "m3_over_gram": y_m3 / y_mean,
                    "s_lin": s_lin,
                })
                print(f"{key:26s} L={L:.4f} y={y_mean:.4f}+-{y_sd:.4f} "
                      f"y_M3={y_m3:.4f} m3/gram={y_m3 / y_mean:.4f}",
                      flush=True)

    np.savez_compressed(os.path.join(DATA, "blind_inputs.npz"), **bank)
    with open(os.path.join(DATA, "blind_predictions.json"), "w") as f:
        json.dump({"ns": NS, "nrep": NREP, "delta_scale": DELTA_SCALE,
                   "operator": "logcumsumexp", "preds": preds}, f, indent=1)
    print("banked", len(preds), "pre-registered predictions")


if __name__ == "__main__":
    main()
