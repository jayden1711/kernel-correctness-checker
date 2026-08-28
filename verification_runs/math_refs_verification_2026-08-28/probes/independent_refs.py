"""Independent verification of math_refs.py (theory-audit flag #8).

THREAT MODEL: math_refs.py was transcribed from the TritonBench reference
kernel sources, and its tests compare against torch built-ins chosen by the
same author under the same assumptions -- a transcription error replicated
into the test would be invisible.

INDEPENDENCE STRATEGY: for every registered operator, this file carries a
from-scratch PURE-PYTHON implementation (Python floats + the math module;
no torch ops anywhere in the computation) derived directly from a fresh
2026-08-28 line-by-line reading of the Triton kernel source -- masked-load
sentinels, loop structure, eps values and placements, reduction denominators,
index arithmetic -- NOT from the math_refs torch expressions and NOT from
torch built-in semantics. For the attention family the emulation reproduces
the kernel's ONLINE-softmax tiling (BLOCK_N=32 running max/normalizer)
rather than the closed form, so agreement also re-proves the streaming
identity at fp64.

Input scales are chosen so that every eps-placement alternative is
numerically distinguishable far above the comparison tolerance:
  - x1e-5 scaled rows make sqrt(s+eps) vs sqrt(s)+eps differ at ~1e-3 rel
    for the 1e-12-eps norms;
  - near-constant rows make layernorm's variance-eps placement dominate.

Pass criterion: max relative deviation <= 1e-10 per operator (fp64
summation-order noise is orders below that at these sizes; any eps/
convention transcription error lands orders above it).
"""

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch  # used ONLY to feed math_refs and read back its outputs

from verification.layer2_numeric_oracle import math_refs as M

REL_TOL = 1e-10
rng = random.Random(20260828)


# ---------------------------------------------------------------- helpers

def randmat(r, c, scale=1.0):
    return [[rng.gauss(0.0, 1.0) * scale for _ in range(c)] for _ in range(r)]


def randvec(n, scale=1.0):
    return [rng.gauss(0.0, 1.0) * scale for _ in range(n)]


def t(x):
    return torch.tensor(x, dtype=torch.float64)


def flatten(o):
    if isinstance(o, (float, int)):
        return [float(o)]
    out = []
    for e in o:
        out.extend(flatten(e))
    return out


def rel_dev(mine, theirs_tensor):
    a = flatten(mine)
    b = [float(v) for v in theirs_tensor.reshape(-1)]
    assert len(a) == len(b), f"shape mismatch: {len(a)} vs {len(b)}"
    denom = max(max(abs(v) for v in b), 1e-300)
    return max(abs(x - y) for x, y in zip(a, b)) / denom


# ------------------------------------------------- pure-python operators
# Each function is written from the Triton kernel source's algorithm.

def p_softmax(x):
    out = []
    for row in x:
        m = max(row)                      # tl.max
        e = [math.exp(v - m) for v in row]
        s = sum(e)                        # tl.sum
        out.append([v / s for v in e])
    return out


def p_log_softmax(x):
    out = []
    for row in x:
        m = max(row)
        s = sum(math.exp(v - m) for v in row)
        out.append([v - m - math.log(s) for v in row])
    return out


def p_gelu(x):
    # y = x * 0.5 * (1 + erf(x * 0.7071067811865476))  [gelu.py]
    return [[v * 0.5 * (1.0 + math.erf(v * 0.7071067811865476)) for v in row] for row in x]


def p_swish(x):
    # sigmoid = 1/(1+exp(-x))  [swish.py]
    return [[v / (1.0 + math.exp(-v)) for v in row] for row in x]


def p_l1norm(x, eps=1e-12):
    # row / (sum|row| + eps) -- eps OUTSIDE, no sqrt  [l1norm.py]
    return [[v / (sum(abs(u) for u in row) + eps) for v in row] for row in x]


def p_l2norm(x, eps=1e-12):
    # row / sqrt(sum(row^2) + eps) -- eps INSIDE sqrt  [l2norm.py]
    return [[v / math.sqrt(sum(u * u for u in row) + eps) for v in row] for row in x]


def p_frobenius(x, eps=1e-12):
    # x / (sqrt(global sumsq) + eps) -- eps OUTSIDE sqrt  [frobenius_norm.py]
    s = sum(v * v for row in x for v in row)
    n = math.sqrt(s) + eps
    return [[v / n for v in row] for row in x]


def p_layernorm(x, gamma, beta, eps=1e-5):
    # mean = sum/n; var = sum((row-mean)^2)/n (biased, masked);
    # (row-mean)/sqrt(var+eps)*gamma+beta  [layernorm.py]
    out = []
    n = len(x[0])
    for row in x:
        mean = sum(row) / n
        var = sum((v - mean) ** 2 for v in row) / n
        inv = 1.0 / math.sqrt(var + eps)
        out.append([(v - mean) * inv * g + b for v, g, b in zip(row, gamma, beta)])
    return out


def p_rmsnorm(x, gamma, eps=1e-5):
    # rms = sqrt(mean(x^2) + eps)  [rmsnorm.py]
    out = []
    n = len(x[0])
    for row in x:
        rms = math.sqrt(sum(v * v for v in row) / n + eps)
        out.append([v / rms * g for v, g in zip(row, gamma)])
    return out


def p_groupnorm(x, num_groups, weight, bias, eps=1e-5):
    # wrapper: x2d = contiguous view (N*G, cpg*spatial); per-row layernorm
    # (no affine), then per-channel gamma/beta  [groupnorm.py]
    N = len(x); C = len(x[0])
    spatial = [len(x[0][0]), len(x[0][0][0])]  # (H, W)
    H, W = spatial
    cpg = C // num_groups
    out = [[[[0.0] * W for _ in range(H)] for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for g in range(num_groups):
            vals = [x[n][g * cpg + cc][h][w]
                    for cc in range(cpg) for h in range(H) for w in range(W)]
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            inv = 1.0 / math.sqrt(var + eps)
            for cc in range(cpg):
                c = g * cpg + cc
                for h in range(H):
                    for w in range(W):
                        out[n][c][h][w] = (x[n][c][h][w] - m) * inv * weight[c] + bias[c]
    return out


def p_instancenorm(x, weight, bias, eps=1e-5):
    # per (n, c) over spatial  [instancenorm.py]
    N = len(x); C = len(x[0]); H = len(x[0][0]); W = len(x[0][0][0])
    out = [[[[0.0] * W for _ in range(H)] for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for c in range(C):
            vals = [x[n][c][h][w] for h in range(H) for w in range(W)]
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            inv = 1.0 / math.sqrt(var + eps)
            for h in range(H):
                for w in range(W):
                    out[n][c][h][w] = (x[n][c][h][w] - m) * inv * weight[c] + bias[c]
    return out


def p_batchnorm(x, rm, rv, weight, bias, eps=1e-5):
    # inference mode, per-channel running stats  [batchnorm.py]
    N = len(x); C = len(x[0]); H = len(x[0][0]); W = len(x[0][0][0])
    out = [[[[0.0] * W for _ in range(H)] for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for c in range(C):
            inv = 1.0 / math.sqrt(rv[c] + eps)
            for h in range(H):
                for w in range(W):
                    out[n][c][h][w] = (x[n][c][h][w] - rm[c]) * inv * weight[c] + bias[c]
    return out


def p_sum(x):
    return [sum(row) for row in x]


def p_mean(x):
    return [sum(row) / len(row) for row in x]


def p_max(x):
    return [max(row) for row in x]


def p_min(x):
    return [min(row) for row in x]


def p_cross_entropy(logits, targets):
    # per-row -log_softmax[target], host mean over rows  [cross_entropy.py]
    losses = []
    for row, tgt in zip(logits, targets):
        m = max(row)
        s = sum(math.exp(v - m) for v in row)
        losses.append(-(row[tgt] - m - math.log(s)))
    return sum(losses) / len(losses)


def p_matmul(a, b):
    Mr, K = len(a), len(a[0])
    Nc = len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(K)) for j in range(Nc)] for i in range(Mr)]


def p_attention_online(q, k, v, causal=False, block_n=32):
    """Online-softmax tile emulation of {flash,sdpa,causal}_attention_kernel:
    running max m, normalizer l, accumulator acc; padded key columns (and,
    when causal, j>i) masked to -inf BEFORE the running-max update."""
    N = len(q); D = len(q[0])
    scale = 1.0 / math.sqrt(D)
    n_tiles = (N + block_n - 1) // block_n
    out = []
    for i in range(N):
        m = -math.inf
        l = 0.0
        acc = [0.0] * D
        for tblk in range(n_tiles):
            start = tblk * block_n
            S = []
            for jj in range(block_n):
                j = start + jj
                if j >= N or (causal and j > i):   # padded / masked column
                    S.append(-math.inf)
                else:
                    S.append(sum(q[i][d] * k[j][d] for d in range(D)) * scale)
            m_new = max(m, max(S))
            r = math.exp(m - m_new) if m_new != -math.inf else 1.0
            acc = [a * r for a in acc]
            P = [math.exp(s - m_new) if s != -math.inf else 0.0 for s in S]
            for jj in range(block_n):
                j = start + jj
                if P[jj] != 0.0:
                    for d in range(D):
                        acc[d] += P[jj] * v[j][d]
            l = l * r + sum(P)
            m = m_new
        out.append([a / l for a in acc])
    return out


def _pool_out(Lin, k, s, p):
    return (Lin + 2 * p - k) // s + 1


def p_avg_pool1d(x, k, s, p):
    # count_include_pad: divide by k always; invalid taps contribute 0  [avg_pool1d.py]
    N = len(x); C = len(x[0]); L = len(x[0][0])
    Lo = _pool_out(L, k, s, p)
    out = [[[0.0] * Lo for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for c in range(C):
            for lo in range(Lo):
                acc = 0.0
                for kk in range(k):
                    li = lo * s - p + kk
                    if 0 <= li < L:
                        acc += x[n][c][li]
                out[n][c][lo] = acc / k
    return out


def p_max_pool1d(x, k, s, p):
    N = len(x); C = len(x[0]); L = len(x[0][0])
    Lo = _pool_out(L, k, s, p)
    out = [[[0.0] * Lo for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for c in range(C):
            for lo in range(Lo):
                best = -math.inf
                for kk in range(k):
                    li = lo * s - p + kk
                    if 0 <= li < L:
                        best = max(best, x[n][c][li])
                out[n][c][lo] = best
    return out


def p_avg_pool2d(x, k, s, p):
    N = len(x); C = len(x[0]); H = len(x[0][0]); W = len(x[0][0][0])
    Ho, Wo = _pool_out(H, k, s, p), _pool_out(W, k, s, p)
    out = [[[[0.0] * Wo for _ in range(Ho)] for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for c in range(C):
            for ho in range(Ho):
                for wo in range(Wo):
                    acc = 0.0
                    for kh in range(k):
                        hi = ho * s - p + kh
                        if not (0 <= hi < H):
                            continue
                        for kw in range(k):
                            wi = wo * s - p + kw
                            if 0 <= wi < W:
                                acc += x[n][c][hi][wi]
                    out[n][c][ho][wo] = acc / (k * k)
    return out


def p_max_pool2d(x, k, s, p):
    N = len(x); C = len(x[0]); H = len(x[0][0]); W = len(x[0][0][0])
    Ho, Wo = _pool_out(H, k, s, p), _pool_out(W, k, s, p)
    out = [[[[0.0] * Wo for _ in range(Ho)] for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for c in range(C):
            for ho in range(Ho):
                for wo in range(Wo):
                    best = -math.inf
                    for kh in range(k):
                        hi = ho * s - p + kh
                        if not (0 <= hi < H):
                            continue
                        for kw in range(k):
                            wi = wo * s - p + kw
                            if 0 <= wi < W:
                                best = max(best, x[n][c][hi][wi])
                    out[n][c][ho][wo] = best
    return out


def p_avg_pool3d(x, k, s, p):
    N = len(x); C = len(x[0]); D = len(x[0][0]); H = len(x[0][0][0]); W = len(x[0][0][0][0])
    Do, Ho, Wo = (_pool_out(D, k, s, p), _pool_out(H, k, s, p), _pool_out(W, k, s, p))
    out = [[[[[0.0] * Wo for _ in range(Ho)] for _ in range(Do)] for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for c in range(C):
            for do in range(Do):
                for ho in range(Ho):
                    for wo in range(Wo):
                        acc = 0.0
                        for kd in range(k):
                            di = do * s - p + kd
                            if not (0 <= di < D):
                                continue
                            for kh in range(k):
                                hi = ho * s - p + kh
                                if not (0 <= hi < H):
                                    continue
                                for kw in range(k):
                                    wi = wo * s - p + kw
                                    if 0 <= wi < W:
                                        acc += x[n][c][di][hi][wi]
                        out[n][c][do][ho][wo] = acc / (k ** 3)
    return out


def p_max_pool3d(x, k, s, p):
    N = len(x); C = len(x[0]); D = len(x[0][0]); H = len(x[0][0][0]); W = len(x[0][0][0][0])
    Do, Ho, Wo = (_pool_out(D, k, s, p), _pool_out(H, k, s, p), _pool_out(W, k, s, p))
    out = [[[[[0.0] * Wo for _ in range(Ho)] for _ in range(Do)] for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for c in range(C):
            for do in range(Do):
                for ho in range(Ho):
                    for wo in range(Wo):
                        best = -math.inf
                        for kd in range(k):
                            di = do * s - p + kd
                            if not (0 <= di < D):
                                continue
                            for kh in range(k):
                                hi = ho * s - p + kh
                                if not (0 <= hi < H):
                                    continue
                                for kw in range(k):
                                    wi = wo * s - p + kw
                                    if 0 <= wi < W:
                                        best = max(best, x[n][c][di][hi][wi])
                        out[n][c][do][ho][wo] = best
    return out


# ------------------------------------------------------------ test cases

def cases():
    """Yields (name, op_name, math_refs args (torch), pure-python result)."""
    # Rowwise, non-pow2 width; ordinary + small-scale (eps discrimination).
    for scale, tag in [(1.0, "randn"), (1e-5, "small_1e-5")]:
        x = randmat(5, 33, scale)
        yield f"softmax/{tag}", "softmax", (t(x),), p_softmax(x)
        yield f"log_softmax/{tag}", "log_softmax", (t(x),), p_log_softmax(x)
        yield f"gelu/{tag}", "gelu", (t(x),), p_gelu(x)
        yield f"swish/{tag}", "swish", (t(x),), p_swish(x)
        yield f"l1norm/{tag}", "l1norm", (t(x),), p_l1norm(x)
        yield f"l2norm/{tag}", "l2norm", (t(x),), p_l2norm(x)
        yield f"frobenius_norm/{tag}", "frobenius_norm", (t(x),), p_frobenius(x)
        yield f"sum_reduction/{tag}", "sum_reduction", (t(x),), p_sum(x)
        yield f"mean_reduction/{tag}", "mean_reduction", (t(x),), p_mean(x)
        yield f"max_reduction/{tag}", "max_reduction", (t(x),), p_max(x)
        yield f"min_reduction/{tag}", "min_reduction", (t(x),), p_min(x)

    # layernorm / rmsnorm: ordinary, small-scale (variance-eps dominates),
    # and near-constant rows (variance ~ 0).
    for scale, tag in [(1.0, "randn"), (1e-3, "small_1e-3")]:
        x = randmat(5, 33, scale)
        g, b = randvec(33), randvec(33)
        yield f"layernorm/{tag}", "layernorm", (t(x), t(g), t(b)), p_layernorm(x, g, b)
        yield f"rmsnorm/{tag}", "rmsnorm", (t(x), t(g)), p_rmsnorm(x, g)
    xc = [[3.0 + rng.gauss(0, 1) * 1e-6 for _ in range(33)] for _ in range(4)]
    g, b = randvec(33), randvec(33)
    yield "layernorm/near_const", "layernorm", (t(xc), t(g), t(b)), p_layernorm(xc, g, b)
    yield "rmsnorm/near_const", "rmsnorm", (t(xc), t(g)), p_rmsnorm(xc, g)

    # groupnorm / instancenorm / batchnorm.
    x4 = [[[[rng.gauss(0, 1) for _ in range(5)] for _ in range(3)] for _ in range(6)] for _ in range(2)]
    w6, b6 = randvec(6), randvec(6)
    yield "groupnorm/G3", "groupnorm", (t(x4), 3, t(w6), t(b6)), p_groupnorm(x4, 3, w6, b6)
    yield "groupnorm/G1", "groupnorm", (t(x4), 1, t(w6), t(b6)), p_groupnorm(x4, 1, w6, b6)
    yield "groupnorm/G6", "groupnorm", (t(x4), 6, t(w6), t(b6)), p_groupnorm(x4, 6, w6, b6)
    yield "instancenorm", "instancenorm", (t(x4), t(w6), t(b6)), p_instancenorm(x4, w6, b6)
    rm, rv = randvec(6), [abs(v) + 0.5 for v in randvec(6)]
    yield "batchnorm", "batchnorm", (t(x4), t(rm), t(rv), t(w6), t(b6)), p_batchnorm(x4, rm, rv, w6, b6)

    # cross_entropy.
    lo = randmat(9, 13)
    tg = [rng.randrange(13) for _ in range(9)]
    yield "cross_entropy", "cross_entropy", (t(lo), torch.tensor(tg)), p_cross_entropy(lo, tg)
    lo_big = randmat(6, 13, 30.0)  # large logits: stability path exercised
    tg2 = [rng.randrange(13) for _ in range(6)]
    yield "cross_entropy/large", "cross_entropy", (t(lo_big), torch.tensor(tg2)), p_cross_entropy(lo_big, tg2)

    # matmul.
    a, b2 = randmat(7, 5), randmat(5, 9)
    yield "matmul", "matmul", (t(a), t(b2)), p_matmul(a, b2)

    # attention family: N=33 exercises the padded second tile (BLOCK_N=32);
    # N=7 exercises a single partial tile; online-softmax emulation.
    for N in (7, 33):
        q, k, v = randmat(N, 8), randmat(N, 8), randmat(N, 8)
        args = (t(q), t(k), t(v))
        yield f"flash_attention/N{N}", "flash_attention", args, p_attention_online(q, k, v)
        yield f"sdpa/N{N}", "scaled_dot_product_attention", args, p_attention_online(q, k, v)
        yield f"causal/N{N}", "causal_flash_attention", args, p_attention_online(q, k, v, causal=True)
    # large-magnitude attention (stability path: running-max rescaling active)
    q, k, v = randmat(33, 8, 6.0), randmat(33, 8, 6.0), randmat(33, 8)
    yield "flash_attention/large", "flash_attention", (t(q), t(k), t(v)), p_attention_online(q, k, v)

    # pools: padding exercised (k=3, s=2, p=1) and pad-free (k=2, s=2, p=0).
    x3 = [[[rng.gauss(0, 1) for _ in range(17)] for _ in range(4)] for _ in range(2)]
    yield "avg_pool1d/p1", "avg_pool1d", (t(x3), 3, 2, 1), p_avg_pool1d(x3, 3, 2, 1)
    yield "avg_pool1d/p0", "avg_pool1d", (t(x3), 2, 2, 0), p_avg_pool1d(x3, 2, 2, 0)
    yield "max_pool1d/p1", "max_pool1d", (t(x3), 3, 2, 1), p_max_pool1d(x3, 3, 2, 1)
    yield "max_pool1d/s1", "max_pool1d", (t(x3), 3, 1, 1), p_max_pool1d(x3, 3, 1, 1)
    x4p = [[[[rng.gauss(0, 1) for _ in range(17)] for _ in range(17)] for _ in range(4)] for _ in range(2)]
    yield "avg_pool2d/p1", "avg_pool2d", (t(x4p), 3, 2, 1), p_avg_pool2d(x4p, 3, 2, 1)
    yield "max_pool2d/p1", "max_pool2d", (t(x4p), 3, 2, 1), p_max_pool2d(x4p, 3, 2, 1)
    x5 = [[[[[rng.gauss(0, 1) for _ in range(9)] for _ in range(9)] for _ in range(9)]
           for _ in range(4)] for _ in range(2)]
    yield "avg_pool3d/p1", "avg_pool3d", (t(x5), 3, 2, 1), p_avg_pool3d(x5, 3, 2, 1)
    yield "max_pool3d/p1", "max_pool3d", (t(x5), 3, 2, 1), p_max_pool3d(x5, 3, 2, 1)


def main():
    results = []
    worst = ("", 0.0)
    ops_seen = set()
    n_fail = 0
    for name, op, args, expect in cases():
        fn = M.get(op)
        assert fn is not None, f"{op} not registered"
        got = fn(*args)
        d = rel_dev(expect, got)
        ok = d <= REL_TOL
        n_fail += (not ok)
        ops_seen.add(op)
        results.append({"case": name, "op": op, "rel_dev": d, "pass": ok})
        if d > worst[1]:
            worst = (name, d)
        print(f"{'PASS' if ok else 'FAIL':4s}  {name:28s}  rel_dev={d:.3e}")

    missing = set(M.registered_ops()) - ops_seen
    print(f"\ncases: {len(results)}  failures: {n_fail}")
    print(f"worst: {worst[0]}  rel_dev={worst[1]:.3e}")
    print(f"registered ops covered: {len(ops_seen)}/{len(M.registered_ops())}"
          + (f"  MISSING: {sorted(missing)}" if missing else ""))

    out = os.path.join(os.path.dirname(__file__), "..", "data", "independent_refs_results.json")
    with open(out, "w") as f:
        json.dump({"cases": results, "n_fail": n_fail,
                   "worst_case": worst[0], "worst_rel_dev": worst[1],
                   "ops_covered": sorted(ops_seen), "ops_missing": sorted(missing)}, f, indent=1)
    sys.exit(1 if (n_fail or missing) else 0)


if __name__ == "__main__":
    main()
