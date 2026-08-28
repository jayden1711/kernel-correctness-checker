"""Per-record adjudication of the 28 norm-family reference-suspect verdicts.

Method (the attention-bug standard): for every record, materialise the
proposal's input exactly, emulate BOTH
    kernel_faithful  -- the TritonBench reference kernel's own arithmetic
                        (layernorm: UNMASKED padded lanes add mean^2 each to
                        the variance sum when n_cols < next_pow2(n_cols);
                        instancenorm: masked, clean; rmsnorm: clean), fp32
    ideal            -- the mathematically correct operator, fp32
then run the SHIPPED check functions through the RUN-ERA wrappers (commit
5277cd1/a4e8aa1: scale/precision wrappers pass the proposal's gamma/beta
through; the identity wrappers use gamma=1/beta=0; the working-tree
_wrap_precision fix postdates the runs).

Adjudication per recorded failed check:
  emulation must reproduce the recorded failure under kernel_faithful
  (validation), and then:
    ideal ALSO fails  -> CHECK-DOMAIN false alarm (eps-vs-variance /
                         cancellation / run-era wrapper bug) -- the reference
                         kernel is not implicated;
    ideal PASSES      -> the failure is caused by the kernel's own arithmetic
                         -> REFERENCE-BUG-CAUSED.
Margins (deviation / atol) are reported so CPU-vs-GPU reduction-order
borderline cases are flagged instead of over-claimed. randn fills have no
recorded seed: 10 seeds, unanimity required, min margin reported.
"""

import json
import math
import os
import torch
import torch.nn.functional as F

import sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
sys.path.insert(0, ROOT)

from verification.layer3_properties import layernorm_properties as lnp
from verification.layer3_properties import rmsnorm_properties as rmp
from verification.layer3_properties import instancenorm_properties as inp

EPS = 1e-5
torch.manual_seed(0)


def npow2(n):
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------- operators
def ln_faithful(x, g, b):
    n = x.shape[-1]
    pad = npow2(n) - n
    mean = x.sum(-1, keepdim=True) / n
    var = ((x - mean) ** 2).sum(-1, keepdim=True) / n + pad * mean ** 2 / n
    return (x - mean) / torch.sqrt(var + EPS) * g + b


def ln_ideal(x, g, b):
    n = x.shape[-1]
    mean = x.sum(-1, keepdim=True) / n
    var = ((x - mean) ** 2).sum(-1, keepdim=True) / n
    return (x - mean) / torch.sqrt(var + EPS) * g + b


def rms_impl(x, g):
    ms = (x * x).sum(-1, keepdim=True) / x.shape[-1]
    return x / torch.sqrt(ms + EPS) * g


def in_impl(x, w, b):
    N, C = x.shape[0], x.shape[1]
    x2 = x.reshape(N * C, -1)
    mean = x2.mean(-1, keepdim=True)
    var = ((x2 - mean) ** 2).mean(-1, keepdim=True)
    out = (x2 - mean) / torch.sqrt(var + EPS)
    out = out.reshape(x.shape)
    shape = [1, C] + [1] * (x.dim() - 2)
    return out * w.view(shape) + b.view(shape)


# ------------------------------------------------- run-era check evaluation
def eval_ln(cand, x, g, b):
    """Returns {check: (ok, margin_or_None, detail)} under run-era wrappers."""
    out = {}
    o_id = cand(x, torch.ones_like(g), torch.zeros_like(b))
    ok, d = lnp.check_zero_mean(o_id)
    out["zero_mean"] = (ok, d)
    ok, d = lnp.check_unit_variance(o_id)
    out["unit_variance"] = (ok, d)
    ok, d = lnp.check_scale_invariance(lambda xi: cand(xi, g, b), x)
    out["scale_invariance"] = (ok, d)
    ok, d = lnp.check_affine_correctness(cand, x)
    out["affine_correctness"] = (ok, d)
    ok, d = lnp.check_precision_coercion(lambda xi: cand(xi, g.to(xi.dtype),
                                                         b.to(xi.dtype)), x)
    out["precision_coercion"] = (ok, d)
    return out


def eval_rms(cand, x, g):
    out = {}
    ok, d = rmp.check_unit_rms(cand(x, torch.ones_like(g)))
    out["unit_rms"] = (ok, d)
    ok, d = rmp.check_scale_invariance(lambda xi: cand(xi, g), x)
    out["scale_invariance"] = (ok, d)
    ok, d = rmp.check_precision_coercion(lambda xi: cand(xi, g.to(xi.dtype)), x)
    out["precision_coercion"] = (ok, d)
    return out


def eval_in(cand, x, w, b):
    out = {}
    o_id = cand(x, torch.ones_like(w), torch.zeros_like(b))
    ok, d = inp.check_zero_mean(o_id)
    out["zero_mean"] = (ok, d)
    ok, d = inp.check_unit_variance(o_id)
    out["unit_variance"] = (ok, d)
    ok, d = inp.check_positive_scale_invariance(cand, x, w, b)
    out["positive_scale_invariance"] = (ok, d)
    return out


# ------------------------------------------------------------------ records
# (pid, op, shape, fill, scale, shift, row_patches, gamma_scale, beta_scale,
#  recorded_failed)
R = [
    ("e678721f", "in", (4, 8, 16, 16), "ones", 5, 0, None, 1, 0, ["unit_variance"]),
    ("65948e8d", "in", (8, 16, 32), "ones", 5, 0, None, 1, 0, ["unit_variance"]),
    ("646a25ef", "in", (4, 8, 16), "ones", 5, 0, None, 1, 0, ["unit_variance"]),
    ("4dcf4ad3", "in", (4, 8, 16, 16), "ones", 5, 0, None, 1, 0, ["unit_variance"]),
    ("66a284c1", "in", (4, 8, 16), "ones", 5, 0, None, 1, 0, ["unit_variance"]),
    ("e93dfc17", "in", (4, 8, 16, 16), "ones", 5, 0, None, 1, 0, ["unit_variance"]),
    ("096858bc", "in", (4, 8, 16), "ones", 1, 0, None, 1, 0, ["unit_variance"]),
    ("f6039549", "in", (4, 8, 16, 16), "ones", 1, 0, None, 1, 0, ["unit_variance"]),
    ("2dc26e8d", "in", (4, 8, 16), "ones", 1, 0, None, 1, 0, ["unit_variance"]),
    ("e7a28a3a", "in", (4, 8, 16, 16), "ones", 1, 0, None, 1, 0, ["unit_variance"]),
    ("e59b42a1", "in", (4, 8, 16, 16), "ones", 1, 0, None, 1, 0, ["unit_variance"]),
    ("55d57718", "in", (8, 16, 32), "ones", 1, 0, None, 1, 0, ["unit_variance"]),
    ("7bfc0f13", "in", (4, 8, 16, 16), "randn", 1e-4, 3, None, 1, 0,
     ["unit_variance", "positive_scale_invariance"]),
    ("bfd36998", "in", (4, 8, 16, 16), "randn", 1e-4, 100, None, 1, 0,
     ["zero_mean", "unit_variance", "positive_scale_invariance"]),
    ("075400e4", "in", (8, 16, 64), "randn", 1e-6, 3, None, 1, 0,
     ["unit_variance", "positive_scale_invariance"]),
    ("2d7f4f3e", "ln", (512, 512), "randn", 0.01, 1000, None, 1, 0,
     ["zero_mean", "unit_variance", "scale_invariance"]),
    ("35205e68", "ln", (512, 512), "randn", 1, 1000, None, 2, 3,
     ["affine_correctness", "precision_coercion"]),
    ("a7e94cf9", "ln", (512, 512), "randn", 1, 0, None, 2, 3,
     ["precision_coercion"]),
    ("f2fac9f6", "ln", (512, 512), "ones", 10, 0,
     [1.0, 5.0, 20.0, -10.0, 100.0, -50.0, 0.0, 7.0, 3.0, -3.0, 15.0, -15.0,
      50.0, -25.0, 8.0, -8.0], 2, 3, ["unit_variance", "precision_coercion"]),
    ("15b63912", "ln", (512, 512), "randn", 0.01, 100, None, 1, 0,
     ["zero_mean", "unit_variance", "scale_invariance"]),
    ("3b2a37a6", "ln", (256, 512), "randn", 0.1, 0, None, 2, 3,
     ["unit_variance", "scale_invariance", "precision_coercion"]),
    ("d8eb4716", "ln", (512, 512), "randn", 1, 100, None, 2, 3,
     ["precision_coercion"]),
    ("16c0b6eb", "ln", (512, 512), "ones", 1, 0,
     [1.0, 2.0, 3.0, -1.0, 4.0, -2.0, 0.0, 1.5, 2.5, -0.5, 3.5, -1.5, 5.0,
      -3.0, 0.5, -0.5], 1, 0, ["unit_variance"]),
    ("a7867d2c", "ln", (256, 512), "randn", 0.01, 0, None, 2, 3,
     ["unit_variance", "scale_invariance", "precision_coercion"]),
    ("f322abe4", "ln", (512, 333), "randn", 1, 10, None, 2, 3,
     ["unit_variance", "affine_correctness", "precision_coercion"]),
    ("a47a431b", "ln", (64, 64), "randn", 1, 0,
     [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0, 0.5, -0.5,
      1.5, -1.5, 2.5], 1, 0, ["unit_variance"]),
    ("0eb28274", "rms", (512, 512), "randn", 1e-8, 0, None, 1, None,
     ["unit_rms", "scale_invariance", "precision_coercion"]),
    ("dc510d61", "rms", (512, 512), "randn", 1, 0, None, 2, None,
     ["precision_coercion"]),
]


def materialize(rec, seed):
    pid, op, shape, fill, scale, shift, patches, gs, bs, _ = rec
    gen = torch.Generator().manual_seed(seed)
    if fill == "randn":
        x = torch.randn(*shape, generator=gen) * scale + shift
    elif fill == "ones":
        x = torch.ones(*shape) * scale + shift
    else:
        x = torch.zeros(*shape) + shift
    if patches is not None:
        for i, v in enumerate(patches):
            x[i, :] = v
    if op == "ln":
        n = shape[-1]
        g = torch.ones(n) * gs
        b = torch.ones(n) * bs if bs else torch.zeros(n)
        return x, g, b
    if op == "rms":
        return x, torch.ones(shape[-1]) * gs, None
    C = shape[1]
    return x, torch.ones(C), torch.zeros(C)


def run_record(rec, seed):
    pid, op, shape, fill, scale, shift, patches, gs, bs, recorded = rec
    x, g, b = materialize(rec, seed)
    if op == "ln":
        faith = eval_ln(ln_faithful, x, g, b)
        ideal = eval_ln(ln_ideal, x, g, b)
    elif op == "rms":
        faith = eval_rms(rms_impl, x, g)
        ideal = faith          # rmsnorm kernel arithmetic == ideal (masked)
    else:
        faith = eval_in(in_impl, x, g, b)
        ideal = faith          # instancenorm kernel masks its lanes: clean
    return faith, ideal


def main():
    results = []
    for rec in R:
        pid, op, shape, fill = rec[0], rec[1], rec[2], rec[3]
        recorded = set(rec[9])
        seeds = range(10) if fill == "randn" else [0]
        per_seed = []
        for sd in seeds:
            faith, ideal = run_record(rec, sd)
            failed_f = {k for k, v in faith.items() if not v[0]}
            failed_i = {k for k, v in ideal.items() if not v[0]}
            per_seed.append((failed_f, failed_i, faith, ideal))
        # validation: does kernel-faithful reproduce the recorded failed set?
        repro = [recorded == f for f, _, _, _ in per_seed]
        # adjudication per recorded failed check
        verdicts = {}
        for chk in sorted(recorded):
            f_all = all(chk in f for f, _, _, _ in per_seed)
            i_all = all(chk in i for _, i, _, _ in per_seed)
            i_none = all(chk not in i for _, i, _, _ in per_seed)
            if not f_all:
                verdicts[chk] = "NOT-REPRODUCED (CPU)"
            elif i_all:
                verdicts[chk] = "check-domain false alarm"
            elif i_none:
                verdicts[chk] = "REFERENCE-BUG-CAUSED"
            else:
                verdicts[chk] = "seed-dependent (mixed)"
        f0, i0, faith0, ideal0 = per_seed[0]
        results.append(dict(pid=pid, op=op, shape=list(shape),
                            recorded=sorted(recorded),
                            reproduced=f"{sum(repro)}/{len(repro)}",
                            emulated_failed=sorted(f0),
                            ideal_failed=sorted(i0),
                            verdicts=verdicts,
                            detail={k: faith0[k][1] for k in sorted(recorded)
                                    if k in faith0}))
        vs = "; ".join(f"{k}={v}" for k, v in verdicts.items())
        print(f"{pid} {op:3s} {str(shape):16s} repro {sum(repro)}/{len(repro)}"
              f"  {vs}")
        for k in sorted(recorded):
            print(f"    {k}: faithful[{faith0[k][1][:78]}]")
            if ideal0[k][1] != faith0[k][1]:
                print(f"    {k}: ideal   [{ideal0[k][1][:78]}]")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "../data/norm_adjudication.json")
    json.dump(results, open(out, "w"), indent=1)
    n_fa = sum(1 for r in results
               if all(v == "check-domain false alarm" for v in r["verdicts"].values()))
    n_bug = sum(1 for r in results
                if any(v == "REFERENCE-BUG-CAUSED" for v in r["verdicts"].values()))
    print(f"\n{len(results)} records: {n_fa} pure check-domain false alarms, "
          f"{n_bug} with reference-bug-caused failures, "
          f"{len(results)-n_fa-n_bug} mixed/other")


if __name__ == "__main__":
    main()
