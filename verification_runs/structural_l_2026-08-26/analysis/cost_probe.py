"""
Cost of the STRUCTURAL path against the MONTE-CARLO path it would replace.

The decisive question this pass has to answer is not "is the closed form
right" -- the generalization round settled that -- but "is computing it
cheaper than probing". Nothing in the prior work measured that, and the
arithmetic below is not obviously favourable: the structural path removes
`n_samples` reference launches and adds `nsim * n_samples * len(profile)`
Gaussian draws.

DEVICE. Measured on CPU, because that is what is available; the GPU that
carried every prior measurement was a rented Colab T4 that is stopped. This
biases the comparison IN FAVOUR of the structural path, and that is why the
result is still usable: a CPU torch reference launch is far slower than the
T4 Triton launch the checker actually pays (banked at 0.1218 ms/sample/call),
so the Monte-Carlo arm is handicapped here. If the structural path loses on
CPU it loses by more on the GPU.

Also reported, and device-independent: the exact RNG draw counts of the two
paths. That ratio is arithmetic, not a measurement, and it does not depend on
which hardware runs it.
"""
import math, os, sys, time, json
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn.functional as F
from verification.layer2_numeric_oracle import structural_l as S

torch.manual_seed(0)
DEV = "cpu"
NS = 20          # the checker's shipped default
NSIM = 3000      # the derivation's NSIM
DELTA = 1e-3

# Corpus-matched shapes. `m` is the row count the generalization round
# reported per operator, so these reproduce its profile sizes.
def build():
    C = []
    x2d = torch.randn(512, 512, device=DEV)
    C.append(("sum_reduction",  torch.randn(64, 128), (), lambda x: x.sum(-1)))
    C.append(("mean_reduction", torch.randn(64, 128), (), lambda x: x.mean(-1)))
    C.append(("max_reduction",  torch.randn(64, 128), (), lambda x: x.max(-1).values))
    C.append(("min_reduction",  torch.randn(64, 128), (), lambda x: x.min(-1).values))
    C.append(("max_pool1d", torch.randn(4, 12, 96), (4, 4, 0), lambda x: F.max_pool1d(x, 4, 4, 0)))
    C.append(("avg_pool1d", torch.randn(4, 12, 96), (4, 4, 0), lambda x: F.avg_pool1d(x, 4, 4, 0)))
    C.append(("max_pool2d", torch.randn(4, 6, 32, 32), (4, 4, 0), lambda x: F.max_pool2d(x, 4, 4, 0)))
    C.append(("avg_pool2d", torch.randn(4, 6, 32, 32), (4, 4, 0), lambda x: F.avg_pool2d(x, 4, 4, 0)))
    C.append(("max_pool3d", torch.randn(2, 3, 16, 16, 16), (2, 2, 0), lambda x: F.max_pool3d(x, 2, 2, 0)))
    C.append(("avg_pool3d", torch.randn(2, 3, 16, 16, 16), (2, 2, 0), lambda x: F.avg_pool3d(x, 2, 2, 0)))
    B = torch.randn(64, 64)
    C.append(("matmul", torch.randn(16, 64), (B,), lambda x, B=B: x @ B))
    C.append(("gelu", torch.randn(64, 128), (), lambda x: F.gelu(x)))
    C.append(("swish", torch.randn(64, 128), (), lambda x: F.silu(x)))
    C.append(("softmax", torch.randn(64, 128), (), lambda x: torch.softmax(x, -1)))
    C.append(("log_softmax", torch.randn(64, 128), (), lambda x: torch.log_softmax(x, -1)))
    C.append(("l1norm", torch.randn(64, 128), (), lambda x: x / x.abs().sum(-1, keepdim=True)))
    C.append(("l2norm", torch.randn(64, 128), (), lambda x: x / x.norm(dim=-1, keepdim=True)))
    C.append(("frobenius_norm", torch.randn(64, 128), (), lambda x: x / x.norm()))
    g = torch.ones(128); b = torch.zeros(128)
    C.append(("layernorm", torch.randn(64, 128), (g, b), lambda x, g=g, b=b: F.layer_norm(x, (128,), g, b)))
    C.append(("rmsnorm", torch.randn(64, 128), (g,),
              lambda x, g=g: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * g))
    gw = torch.ones(8); gb = torch.zeros(8)
    C.append(("groupnorm", torch.randn(4, 8, 16, 16), (2, gw, gb),
              lambda x, gw=gw, gb=gb: F.group_norm(x, 2, gw, gb)))
    C.append(("instancenorm", torch.randn(4, 8, 16, 16), (gw, gb),
              lambda x, gw=gw, gb=gb: F.instance_norm(x, weight=gw, bias=gb)))
    rm = torch.zeros(8); rv = torch.ones(8)
    C.append(("batchnorm", torch.randn(4, 8, 16, 16), (rm, rv, gw, gb),
              lambda x, rm=rm, rv=rv, gw=gw, gb=gb:
                  F.batch_norm(x, rm, rv, gw, gb, False, 0.1, 1e-5)))
    tg = torch.randint(0, 100, (64,))
    C.append(("cross_entropy", torch.randn(64, 100), (tg,),
              lambda x, tg=tg: F.cross_entropy(x, tg)))
    K = torch.randn(64, 32); V = torch.randn(64, 32)
    C.append(("flash_attention", torch.randn(64, 32), (K, V),
              lambda q, K=K, V=V: F.scaled_dot_product_attention(q, K, V)))
    C.append(("causal_flash_attention", torch.randn(64, 32), (K, V),
              lambda q, K=K, V=V: F.scaled_dot_product_attention(q, K, V, is_causal=True)))
    C.append(("scaled_dot_product_attention", torch.randn(64, 32), (K, V),
              lambda q, K=K, V=V: F.scaled_dot_product_attention(q, K, V)))
    return C


def timeit(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1000.0


def mc_arm(ref, x, x_std):
    """Exactly what _probe_adaptive_tol_and_sens does, unbatched."""
    base = ref(x)
    sens = []
    for _ in range(NS):
        d = torch.randn_like(x) * DELTA * x_std
        sens.append((ref(x + d) - base).abs().max())
    st = torch.stack(sens).to(device="cpu", dtype=torch.float32)
    return max(3.0 * torch.quantile(st, 0.95).item(), 1e-6)


def main():
    rows = []
    print(f"{'operator':<32}{'m':>7}{'MC ms':>9}{'struct ms':>11}"
          f"{'  =prof':>9}{'+M3':>9}{'ratio':>9}{'MC draws':>12}{'M3 draws':>13}{'x':>8}")
    for op, x, rest, ref in build():
        x = x.to(DEV)
        x_std = x.float().std().item()
        t_mc = timeit(lambda: mc_arm(ref, x, x_std))
        rn = S.row_norms(op, x, list(rest))
        m = rn.numel()
        t_prof = timeit(lambda: S.row_norms(op, x, list(rest)))
        t_m3 = timeit(lambda: S.y_profile(rn.float(), NS, nsim=NSIM), reps=1)
        t_st = t_prof + t_m3
        w = rn[rn > 0]
        wn = w.numel()
        if wn > 30000:
            wn = min(wn, 30000)   # the cap in y_profile
        mc_draws = NS * x.numel()
        m3_draws = NSIM * NS * wn
        rows.append(dict(op=op, m=m, mc_ms=t_mc, prof_ms=t_prof, m3_ms=t_m3,
                         struct_ms=t_st, mc_draws=mc_draws, m3_draws=m3_draws))
        print(f"{op:<32}{m:>7}{t_mc:>9.2f}{t_st:>11.1f}{t_prof:>9.2f}{t_m3:>9.1f}"
              f"{t_st/t_mc:>9.1f}{mc_draws:>12,}{m3_draws:>13,}"
              f"{m3_draws/mc_draws:>8.1f}")

    print()
    tot_mc = sum(r["mc_ms"] for r in rows)
    tot_st = sum(r["struct_ms"] for r in rows)
    stat = [r for r in rows if r["op"] in S.STATIC_OPS]
    dyn = [r for r in rows if r["op"] not in S.STATIC_OPS]
    print(f"ALL 27 ops   MC {tot_mc:8.1f} ms   structural {tot_st:9.1f} ms"
          f"   -> {tot_st/tot_mc:.1f}x")
    for nm, grp in (("shape-only (9)", stat), ("input-dependent (18)", dyn)):
        a = sum(r["mc_ms"] for r in grp); b = sum(r["struct_ms"] for r in grp)
        print(f"  {nm:<22} MC {a:8.1f} ms   structural {b:9.1f} ms   -> {b/a:.1f}x")
    print()
    tot_mcd = sum(r["mc_draws"] for r in rows)
    tot_m3d = sum(r["m3_draws"] for r in rows)
    print(f"RNG draws (device-independent arithmetic):")
    print(f"  Monte-Carlo path : {tot_mcd:>15,}")
    print(f"  structural path  : {tot_m3d:>15,}   = {tot_m3d/tot_mcd:.0f}x more")

    json.dump(rows, open("verification_runs/structural_l_2026-08-26/analysis/cost.json", "w"), indent=1)


if __name__ == "__main__":
    main()
