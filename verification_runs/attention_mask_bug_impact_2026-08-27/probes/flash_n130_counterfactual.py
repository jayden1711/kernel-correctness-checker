"""The three N=130 flash_attention adversarial-search proposals: recorded-world
validation and corrected-reference counterfactual.

Recorded (run 7b0a6e8d, 2026-07-23): all three verdicts say
"Reference failed: ['attention_weights_sum_to_one']" -- the checker catching
the reference kernel's own padded-column bug live (30 padded columns at
N=130), booked as "invalid input". Proposal idx 9 (w0, iter 2) additionally
recorded a gap-confirmed skip_rescaling catch, three seconds BEFORE the run's
recorded winning hit (idx 10, N=96, approx_denom).

This probe emulates, in fp32 tile-faithful CPU math:
  ref_buggy       shipped reference (scale 1/sqrt(D), padded S=0 columns)
  ref_correct     true attention
  four mutants    skip_rescaling / approx_denom / drop_last_tile / wrong_mask,
                  each per its own source (all lack the 1/sqrt(D) scale;
                  padding as their loops actually produce it)

and reports, per proposal:
  (a) recorded-world validation: ref_buggy's weights-sum deviation (must
      exceed the check's atol=1e-3, matching the recorded reference-fail),
      and for idx 9 the recorded skip_rescaling gap;
  (b) corrected-world counterfactual: ref_correct passes weights-sum?; per
      mutant, weights-sum deviation (an emulatable checker catch) and naive
      allclose vs ref_correct (atol=1e-3, rtol=1e-2) -- is_hit lower bound
      = ref valid AND (catch AND naive-pass) for some mutant.

randn fills have no recorded seed; those proposals are run over 10 seeds and
conclusions reported as counts. Proposal idx 9 is fully deterministic.
"""

import math
import os
import torch

torch.manual_seed(0)
N, Dh = 130, 64
BN, NP = 32, 160          # BLOCK_N, padded length
NAIVE = dict(atol=1e-3, rtol=1e-2)


def build(spec, gen):
    fill = spec["fill"]
    if fill == "randn":
        t = torch.randn(N, Dh, generator=gen)
    elif fill == "zeros":
        t = torch.zeros(N, Dh)
    elif fill == "ones":
        t = torch.ones(N, Dh)
    t = t * spec["scale"] + spec["shift"]
    for p in spec["patches"]:
        assert p["indices"] == "[128:, :]"
        t[128:, :] = p["value"]
    return t


PROPS = {
    1: dict(Q=dict(fill="randn", scale=1.0, shift=0, patches=[]),
            K=dict(fill="randn", scale=1.0, shift=0,
                   patches=[dict(indices="[128:, :]", value=1e4)]),
            V=dict(fill="randn", scale=1.0, shift=0,
                   patches=[dict(indices="[128:, :]", value=1e4)]),
            random=True),
    6: dict(Q=dict(fill="randn", scale=0.1, shift=0, patches=[]),
            K=dict(fill="zeros", scale=1.0, shift=0,
                   patches=[dict(indices="[128:, :]", value=10.0)]),
            V=dict(fill="zeros", scale=1.0, shift=0,
                   patches=[dict(indices="[128:, :]", value=5.0)]),
            random=True),
    9: dict(Q=dict(fill="ones", scale=0.01, shift=0, patches=[]),
            K=dict(fill="zeros", scale=1.0, shift=0,
                   patches=[dict(indices="[128:, :]", value=0.5)]),
            V=dict(fill="zeros", scale=1.0, shift=0,
                   patches=[dict(indices="[128:, :]", value=1.0)]),
            random=False),
}


def pad_kv(K, V):
    Kp = torch.cat([K, K.new_zeros(NP - N, Dh)])
    Vp = torch.cat([V, V.new_zeros(NP - N, Dh)])
    return Kp, Vp


def online(Q, Kp, Vp, scale, tiles, l_update, S_mask=None):
    """Faithful online-softmax tile loop in fp32."""
    M = Q.shape[0]
    m = torch.full((M,), float("-inf"))
    l = torch.zeros(M)
    acc = torch.zeros(M, Dh)
    for ti, start in enumerate(tiles):
        Kb = Kp[start:start + BN]
        Vb = Vp[start:start + BN]
        S = (Q @ Kb.T) * scale
        if S_mask is not None:
            S = S_mask(S, start)
        m_new = torch.maximum(m, S.max(dim=1).values)
        e = torch.exp(m - m_new)
        e = torch.where(torch.isnan(e), torch.zeros_like(e), e)  # -inf - -inf guard
        acc = acc * e.unsqueeze(1)
        P = torch.exp(S - m_new.unsqueeze(1))
        acc = acc + P @ Vb
        l = l_update(l, e, P, ti)
        m = m_new
    return acc / l.unsqueeze(1)


def ref_buggy(Q, K, V):
    Kp, Vp = pad_kv(K, V)
    return online(Q, Kp, Vp, 1 / math.sqrt(Dh), range(0, NP, BN),
                  lambda l, e, P, ti: l * e + P.sum(1))


def ref_correct(Q, K, V):
    S = (Q @ K.T) / math.sqrt(Dh)
    return torch.softmax(S, dim=-1) @ V


def mut_skip_rescaling(Q, K, V):
    Kp, Vp = pad_kv(K, V)
    # no scale, acc never rescaled, l accumulated without rescale
    M = Q.shape[0]
    m = torch.full((M,), float("-inf"))
    l = torch.zeros(M)
    acc = torch.zeros(M, Dh)
    for start in range(0, NP, BN):
        Kb, Vb = Kp[start:start + BN], Vp[start:start + BN]
        S = Q @ Kb.T
        m_new = torch.maximum(m, S.max(dim=1).values)
        P = torch.exp(S - m_new.unsqueeze(1))
        acc = acc + P @ Vb
        l = l + P.sum(1)
        m = m_new
    return acc / l.unsqueeze(1)


def mut_approx_denom(Q, K, V):
    Kp, Vp = pad_kv(K, V)
    half = (NP // BN) // 2
    return online(Q, Kp, Vp, 1.0, range(0, NP, BN),
                  lambda l, e, P, ti: (l * e + P.sum(1)) if ti < half else l * e)


def mut_drop_last_tile(Q, K, V):
    Kp, Vp = pad_kv(K, V)
    # range(0, N - BN, BN) with N the TRUE N=130: starts 0,32,64,96
    return online(Q, Kp, Vp, 1.0, range(0, N - BN, BN),
                  lambda l, e, P, ti: l * e + P.sum(1))


def mut_wrong_mask(Q, K, V):
    Kp, Vp = pad_kv(K, V)

    def smask(S, start):
        qi = torch.arange(S.shape[0]).unsqueeze(1)
        ki = torch.arange(start, start + BN).unsqueeze(0)
        return S.masked_fill(~(qi > ki + 1), float("-inf"))
    return online(Q, Kp, Vp, 1.0, range(0, NP, BN),
                  lambda l, e, P, ti: l * e + P.sum(1), S_mask=smask)


MUTS = dict(skip_rescaling=mut_skip_rescaling, approx_denom=mut_approx_denom,
            drop_last_tile=mut_drop_last_tile, wrong_mask=mut_wrong_mask)


def wsum_dev(fn, Q, K, V):
    out = fn(Q, K, torch.ones_like(V))
    if not torch.isfinite(out).all():
        return float("inf")
    return (out - torch.ones_like(out)).abs().max().item()


def analyze(idx, seed):
    gen = torch.Generator().manual_seed(seed)
    p = PROPS[idx]
    Q, K, V = build(p["Q"], gen), build(p["K"], gen), build(p["V"], gen)
    dev_bug = wsum_dev(ref_buggy, Q, K, V)
    dev_cor = wsum_dev(ref_correct, Q, K, V)
    rc = ref_correct(Q, K, V)
    res = dict(dev_bug=dev_bug, dev_cor=dev_cor, muts={})
    for name, fn in MUTS.items():
        mo = fn(Q, K, V)
        caught = wsum_dev(fn, Q, K, V) > 1e-3 or not torch.isfinite(mo).all()
        naive = torch.isfinite(mo).all() and torch.allclose(
            mo.float(), rc.float(), **NAIVE)
        res["muts"][name] = (caught, bool(naive))
    return res


for idx in [1, 6, 9]:
    seeds = range(10) if PROPS[idx]["random"] else [0]
    hits = 0
    ref_fail_bug = ref_pass_cor = 0
    mut_stats = {}
    for sd in seeds:
        r = analyze(idx, sd)
        ref_fail_bug += r["dev_bug"] > 1e-3
        ref_pass_cor += r["dev_cor"] <= 1e-3
        hit = r["dev_cor"] <= 1e-3 and any(
            c and nv for c, nv in r["muts"].values())
        hits += hit
        for k, (c, nv) in r["muts"].items():
            a, b = mut_stats.get(k, (0, 0))
            mut_stats[k] = (a + c, b + (c and nv))
    n = len(list(seeds))
    r0 = analyze(idx, 0)
    print(f"proposal idx {idx} ({'random' if PROPS[idx]['random'] else 'deterministic'}, "
          f"{n} draw(s)):")
    print(f"  recorded-world: ref_buggy weights-sum dev {r0['dev_bug']:.4f} "
          f"(fails>{1e-3:g}: {ref_fail_bug}/{n})   [recorded: Reference failed]")
    print(f"  corrected-world: ref passes weights-sum {ref_pass_cor}/{n} "
          f"(dev {r0['dev_cor']:.2e})")
    for k, (c, g) in mut_stats.items():
        print(f"    {k:16s} caught {c}/{n}  caught-with-naive-gap {g}/{n}")
    print(f"  => counterfactual is_hit (lower bound): {hits}/{n}")
