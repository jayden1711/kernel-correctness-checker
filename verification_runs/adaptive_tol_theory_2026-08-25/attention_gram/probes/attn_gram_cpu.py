"""Attention Gram-matrix extension, CPU stage.

Applies theory_audit_2026-08-27 H1's method to the attention family — the
one with the next-worst M3 residual (+8..17%). For attention the perturbation
is applied to Q only, and output row i depends only on Q row i, so the exact
first-order law is

    s/sigma = max_{i,d} | <J_{(i,d)}, z_i> |,   z_i ~ N(0, I_D) iid per row,

i.e. block-diagonal across rows with the full D_v x D_v within-row Gram (the
"rows share a softmax denominator" correlation M3 discards). Here we compute
the EXACT Jacobian (autograd, float64) at the corpus's own replayed inputs
and predict y = tol/(3 sigma L) with zero fitted constants.

Inputs are replayed bit-for-bit: tritonbench_registry's OPS x mutants order,
6 draws per corpus entry from np.random.default_rng(0) — the same stream
gpu_native.py consumed on the T4. The banked sigma is the alignment check:
it must match 1e-3 * std(Q) to float precision, or the replay is wrong.

Measured y comes from the banked native run (gpu_native.jsonl, real Triton
kernels). L in y uses the exact Jacobian's max row norm, not the banked
K=400 MC estimate (which is biased high ~8-12%; the ratio is reported as a
second alignment check).

Outputs: data/attn_gram_cpu.json, data/qkv_corpus.npz (banked Q/K/V).
"""

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../../../..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks/autokernel/files"))

from tritonbench_registry import OPS, FAMILIES  # numpy/torch only at import

NATIVE = os.path.join(HERE, "../../native_run/gpu_native.jsonl")
GEN = os.path.join(HERE, "../../generalization/data/gen_native.jsonl")
ATT_OPS = {"causal_flash_attention", "flash_attention",
           "scaled_dot_product_attention"}
NS = 40
NREP = 800
torch.manual_seed(0)


def ref_fn(op):
    def flash(Q, K, V):
        S = (Q @ K.T) / math.sqrt(Q.shape[1])
        return torch.softmax(S, dim=-1) @ V

    def causal(Q, K, V):
        N = Q.shape[0]
        S = (Q @ K.T) / math.sqrt(Q.shape[1])
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))
        return torch.softmax(S, dim=-1) @ V
    return causal if op == "causal_flash_attention" else flash


def replay_inputs():
    """Reproduce the exact per-(entry, inv) numpy inputs of the native run."""
    rng = np.random.default_rng(0)
    out = {}
    entry = 0
    for spec_key, ref_file, cheat_dir, family, mutant_names in OPS:
        mk_fn = FAMILIES[family][0]
        for _mut in mutant_names:
            for j in range(6):
                np_args = mk_fn(rng)
                if spec_key in ATT_OPS:
                    out[(entry, j)] = (spec_key, np_args)
            entry += 1
    return out


def jacobian_rows(op, Qn, Kn, Vn):
    """Exact J (m x N*D) of the reference w.r.t. Q, float64."""
    f = ref_fn(op)
    Q = torch.from_numpy(Qn).double().requires_grad_(True)
    K = torch.from_numpy(Kn).double()
    V = torch.from_numpy(Vn).double()
    J = torch.autograd.functional.jacobian(lambda q: f(q, K, V), Q,
                                           vectorize=False)
    m = J.shape[0] * J.shape[1]
    return J.reshape(m, Q.numel())


def q95_torch(v):
    v, _ = torch.sort(v, dim=-1)
    n = v.shape[-1]
    h = 0.95 * (n - 1)
    lo = int(math.floor(h))
    fr = h - lo
    return v[..., lo] * (1 - fr) + v[..., lo + 1] * fr


def sim_exact(J32, L, nrep=NREP, chunk=4000):
    """y-hat = q95_40( max |J z| ) / L over nrep replications."""
    tot = []
    need = nrep * NS
    dim = J32.shape[1]
    done = 0
    while done < need:
        b = min(chunk, need - done)
        z = torch.randn(b, dim)
        s = (z @ J32.T).abs().amax(dim=1)
        tot.append(s)
        done += b
    s = torch.cat(tot).reshape(nrep, NS)
    y = q95_torch(s) / L
    return y.mean().item(), y.std().item()


def sim_m3(rn32, nrep=NREP, chunk=4000):
    """M3's orthogonal-rows assumption: independent z per output coord."""
    L = rn32.max()
    w = (rn32 / L)
    tot = []
    need = nrep * NS
    done = 0
    while done < need:
        b = min(chunk, need - done)
        z = torch.randn(b, w.numel()).abs()
        s = (z * w).amax(dim=1)
        tot.append(s)
        done += b
    s = torch.cat(tot).reshape(nrep, NS)
    y = q95_torch(s)
    return y.mean().item(), y.std().item()


def main():
    inputs = replay_inputs()
    recs = [json.loads(l) for l in open(NATIVE)]
    prim = {(r["entry"], r["inv"]): r for r in recs
            if r["op"] in ATT_OPS and r["kind"] == "primary"}
    gen = {r["op"]: r for r in map(json.loads, open(GEN)) if r["op"] in ATT_OPS}

    results = []
    qkv_bank = {}
    seen_flash_entries = set()
    for (entry, inv), (op, np_args) in sorted(inputs.items()):
        key = (entry, inv)
        if key not in prim:
            continue
        # flash_attention appears once per mutant entry (4 entries); the
        # measurement is of the same reference kernel each time, but on
        # DIFFERENT drawn inputs -- keep all, they are extra test points.
        r = prim[key]
        Qn, Kn, Vn = np_args
        sig_replay = 1e-3 * torch.from_numpy(Qn).float().std().item()
        sig_err = abs(sig_replay - r["sigma"]) / r["sigma"]

        J = jacobian_rows(op, Qn, Kn, Vn)
        rn = J.norm(dim=1)
        L = rn.max().item()
        y_meas = r["tol"] / (3 * r["sigma"] * L)
        J32 = J.float()
        y_pred, y_sd = sim_exact(J32, L)
        m3_pred, _ = sim_m3(rn.float())

        results.append(dict(
            op=op, entry=entry, inv=inv, sigma_replay_err=sig_err,
            L_exact=L, L_mc400_ratio=r["L"] / L,
            tol=r["tol"], y_meas=y_meas, y_pred=y_pred, y_sd=y_sd,
            z=(y_meas - y_pred) / y_sd,
            m3_pred=m3_pred, m3_over_gram=m3_pred / y_pred,
            m3_over_meas=m3_pred / y_meas,
            defect_t01=r.get("defect_t01"),
            y_M3_banked=gen.get(op, {}).get("y_M3") if inv == 0 else None,
        ))
        qkv_bank[f"{op}_{entry}_{inv}_Q"] = Qn
        qkv_bank[f"{op}_{entry}_{inv}_K"] = Kn
        qkv_bank[f"{op}_{entry}_{inv}_V"] = Vn
        print(f"{op:30s} e{entry} inv{inv} sig_err {sig_err:.2e} "
              f"Lmc/Lex {r['L']/L:.3f}  y_meas {y_meas:.4f} "
              f"y_pred {y_pred:.4f}+-{y_sd:.4f} z {(y_meas-y_pred)/y_sd:+.2f} "
              f" m3/gram {m3_pred/y_pred:.3f} m3/meas {m3_pred/y_meas:.3f}",
              flush=True)

    np.savez_compressed(os.path.join(HERE, "../data/qkv_corpus.npz"), **qkv_bank)
    json.dump(results, open(os.path.join(HERE, "../data/attn_gram_cpu.json"), "w"),
              indent=1)

    print("\n=== per-op summary ===")
    for op in sorted(ATT_OPS):
        rs = [r for r in results if r["op"] == op]
        if not rs:
            continue
        mp = np.array([r["y_meas"] / r["y_pred"] for r in rs])
        m3g = np.array([r["m3_over_gram"] for r in rs])
        m3m = np.array([r["m3_over_meas"] for r in rs])
        zs = np.array([r["z"] for r in rs])
        print(f"{op:30s} n={len(rs):2d} meas/pred median {np.median(mp):.4f} "
              f"[{mp.min():.4f},{mp.max():.4f}]  worst|z| {abs(zs).max():.2f}  "
              f"m3/gram median {np.median(m3g):.3f}  m3/meas median {np.median(m3m):.3f}")
    ge = [r for r in results if r["inv"] == 0 and r["y_M3_banked"]]
    if ge:
        print("\ny_M3 anchor (banked gen_native vs this sim, j=0 entries):")
        for r in ge:
            print(f"  {r['op']:30s} banked {r['y_M3_banked']:.4f} "
                  f"sim {r['m3_pred']:.4f} ratio {r['y_M3_banked']/r['m3_pred']:.4f}")


if __name__ == "__main__":
    main()
