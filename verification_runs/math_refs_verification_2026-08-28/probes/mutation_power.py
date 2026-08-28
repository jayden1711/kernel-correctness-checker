"""Power check for independent_refs.py: inject the transcription errors the
flag worries about into the PURE-PYTHON side and measure the deviation each
produces against math_refs under the same test inputs. Every mutation must
land far above REL_TOL=1e-10 on at least one existing case, or the main
probe's clean pass would be uninformative for that error class."""

import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import independent_refs as IR  # noqa: E402  (same directory)

sys.path.insert(0, os.path.dirname(__file__))


def dev(op, args, mutated):
    from verification.layer2_numeric_oracle import math_refs as M
    return IR.rel_dev(mutated, M.get(op)(*args))


def main():
    rng = random.Random(7)
    randmat = IR.randmat
    t = IR.t
    rows = []

    # 1. l2norm eps OUTSIDE sqrt instead of inside (small-scale input).
    x = randmat(5, 33, 1e-5)
    mut = [[v / (math.sqrt(sum(u * u for u in row)) + 1e-12) for v in row] for row in x]
    rows.append(("l2norm eps outside sqrt", dev("l2norm", (t(x),), mut)))

    # 2. frobenius eps INSIDE sqrt instead of outside (small-scale input).
    s = sum(v * v for row in x for v in row)
    n = math.sqrt(s + 1e-12)
    mut = [[v / n for v in row] for row in x]
    rows.append(("frobenius eps inside sqrt", dev("frobenius_norm", (t(x),), mut)))

    # 3. layernorm UNBIASED variance (n-1) on ordinary input.
    x = randmat(5, 33)
    g, b = IR.randvec(33), IR.randvec(33)
    nA = 33
    mut = []
    for row in x:
        m = sum(row) / nA
        var = sum((v - m) ** 2 for v in row) / (nA - 1)
        inv = 1.0 / math.sqrt(var + 1e-5)
        mut.append([(v - m) * inv * gg + bb for v, gg, bb in zip(row, g, b)])
    rows.append(("layernorm unbiased var", dev("layernorm", (t(x), t(g), t(b)), mut)))

    # 4. layernorm eps=1e-6 instead of 1e-5 (near-constant rows).
    xc = [[3.0 + rng.gauss(0, 1) * 1e-6 for _ in range(33)] for _ in range(4)]
    mut = []
    for row in xc:
        m = sum(row) / nA
        var = sum((v - m) ** 2 for v in row) / nA
        inv = 1.0 / math.sqrt(var + 1e-6)
        mut.append([(v - m) * inv * gg + bb for v, gg, bb in zip(row, g, b)])
    rows.append(("layernorm eps 1e-6", dev("layernorm", (t(xc), t(g), t(b)), mut)))

    # 5. rmsnorm eps OUTSIDE sqrt (near-constant rows are not needed; use small).
    xs = randmat(5, 33, 1e-3)
    g33 = IR.randvec(33)
    mut = [[v / (math.sqrt(sum(u * u for u in row) / nA) + 1e-5) * gg
            for v, gg in zip(row, g33)] for row in xs]
    rows.append(("rmsnorm eps outside sqrt", dev("rmsnorm", (t(xs), t(g33)), mut)))

    # 6. attention scale 1/D instead of 1/sqrt(D).
    q, k, v = randmat(33, 8), randmat(33, 8), randmat(33, 8)
    orig = IR.p_attention_online

    def scaled(qq, kk, vv, causal=False, block_n=32):
        # recompute with wrong scale by pre-scaling q
        f = 1.0 / math.sqrt(8)  # extra 1/sqrt(D): net 1/D
        q2 = [[e * f for e in row] for row in qq]
        return orig(q2, kk, vv, causal=causal, block_n=block_n)

    rows.append(("attention scale 1/D", dev("flash_attention", (t(q), t(k), t(v)), scaled(q, k, v))))

    # 7. causal convention j < i (excludes self) instead of j <= i.
    def causal_strict(qq, kk, vv, block_n=32):
        N, D = len(qq), len(qq[0])
        sc = 1.0 / math.sqrt(D)
        out = []
        for i in range(N):
            S = [sum(qq[i][d] * kk[j][d] for d in range(D)) * sc if j < i else -math.inf
                 for j in range(N)]
            m = max(S) if i > 0 else 0.0
            if i == 0:  # degenerate: no keys visible; kernel would produce nan
                out.append([float("nan")] * D)
                continue
            e = [math.exp(sv - m) if sv != -math.inf else 0.0 for sv in S]
            Z = sum(e)
            out.append([sum(e[j] * vv[j][d] for j in range(N)) / Z for d in range(D)])
        return out

    mut = causal_strict(q, k, v)
    # compare on rows 1.. (row 0 nan under the mutation — already a loud diff)
    from verification.layer2_numeric_oracle import math_refs as M
    ref = M.get("causal_flash_attention")(t(q), t(k), t(v))
    d7 = IR.rel_dev([mut[i] for i in range(1, 33)], ref[1:])
    rows.append(("causal j<i (rows 1..)", d7))

    # 8. avg_pool count_include_pad=False (divide by valid count).
    x3 = [[[rng.gauss(0, 1) for _ in range(17)] for _ in range(4)] for _ in range(2)]
    N, C, L = 2, 4, 17
    kk_, s_, p_ = 3, 2, 1
    Lo = IR._pool_out(L, kk_, s_, p_)
    mut = [[[0.0] * Lo for _ in range(C)] for _ in range(N)]
    for n2 in range(N):
        for c in range(C):
            for lo in range(Lo):
                acc, cnt = 0.0, 0
                for kx in range(kk_):
                    li = lo * s_ - p_ + kx
                    if 0 <= li < L:
                        acc += x3[n2][c][li]
                        cnt += 1
                mut[n2][c][lo] = acc / cnt
    rows.append(("avg_pool1d count_include_pad=False", dev("avg_pool1d", (t(x3), 3, 2, 1), mut)))

    # 9. mean_reduction dividing by BLOCK_SIZE (next pow2 = 64) instead of n_cols.
    x = randmat(5, 33)
    mut = [sum(row) / 64 for row in x]
    rows.append(("mean / next_pow2 width", dev("mean_reduction", (t(x),), mut)))

    # 10. cross_entropy reduction sum instead of mean.
    lo = randmat(9, 13)
    tg = [rng.randrange(13) for _ in range(9)]
    mut = IR.p_cross_entropy(lo, tg) * 9
    rows.append(("cross_entropy sum-reduction", dev("cross_entropy", (t(lo), IR.torch.tensor(tg)), mut)))

    print(f"{'mutation':42s}  rel_dev      detectable(>1e-10)")
    all_ok = True
    for name, d in rows:
        ok = d > 1e-10
        all_ok &= ok
        print(f"{name:42s}  {d:.3e}  {'YES' if ok else 'NO — POWER GAP'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
