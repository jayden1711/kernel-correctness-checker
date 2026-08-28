"""Uncertainty for two reported statistics (theory-audit flag #6, second half).

A. CFA_NONHIT_ROOTCAUSE.md §6: "no-context operators need ~1.9x more
   proposals" — n=9 operators, point estimate, no CI.
B. BUG_CLASS_THEORY.md leakage ablation: 112/120 offline accuracy and 83/120
   validity-term agreement, used comparatively without error bars.

Everything below is exact or seeded-bootstrap, offline, no GPU.
"""

import itertools
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


# ------------------------------------------------------------ helpers

def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((ctr - rad) / den, (ctr + rad) / den)


def mean(xs):
    return sum(xs) / len(xs)


# ============================== A: context effect =====================

def part_a():
    print("=" * 70)
    print("A. context effect on proposals-to-hit")
    print("=" * 70)
    # canonical banked totals (adversarial_results/*_search_result.json)
    ctx = {"softmax": 12, "layernorm": 12, "matmul": 4,
           "flash_attention": 12, "rmsnorm": 4}
    noctx_hit = {"gelu": 18, "instancenorm": 18, "argmax": 24}
    censored = {"causal_flash_attention": 120}  # budget-capped, no hit

    a = list(ctx.values())          # context group
    b = list(noctx_hit.values())    # no-context, hits only (doc's mean basis)
    print(f"context group {sorted(ctx.items())}: mean {mean(a):.1f}")
    print(f"no-context hits {sorted(noctx_hit.items())}: mean {mean(b):.1f}")
    print(f"censored: causal_flash_attention >= 120 (excluded from doc's mean)")
    print(f"ratio of means (banked artifacts): {mean(b)/mean(a):.2f}x"
          f"   [doc reports 20.0/10.8 = 1.9x; 10.8 not reproducible from"
          f" banked artifacts — see FINDINGS]")

    # Exact permutation test on the hit-only 8 operators: under H0 (labels
    # exchangeable), how often does a random 3-subset assigned "no-context"
    # have mean - (other 5)'s mean >= observed?
    vals = a + b
    obs = mean(b) - mean(a)
    count = 0
    total = 0
    for combo in itertools.combinations(range(8), 3):
        g = [vals[i] for i in combo]
        rest = [vals[i] for i in range(8) if i not in combo]
        total += 1
        if mean(g) - mean(rest) >= obs - 1e-12:
            count += 1
    print(f"\nexact permutation test (hit-only, 8 ops, C(8,3)={total}):"
          f" one-sided p = {count}/{total} = {count/total:.4f}")

    # Include the censored operator conservatively AT its budget (120 is a
    # lower bound for its true proposals-to-hit, so this understates the
    # no-context mean and the test is still valid as a lower-bound statement).
    vals9 = a + b + [120]
    obs9 = mean(b + [120]) - mean(a)
    count9 = tot9 = 0
    for combo in itertools.combinations(range(9), 4):
        g = [vals9[i] for i in combo]
        rest = [vals9[i] for i in range(9) if i not in combo]
        tot9 += 1
        if mean(g) - mean(rest) >= obs9 - 1e-12:
            count9 += 1
    print(f"with censored op at its lower bound 120 (C(9,4)={tot9}):"
          f" one-sided p = {count9}/{tot9} = {count9/tot9:.4f}")

    # Bootstrap CI for the ratio of means (hit-only), 10^5 resamples, seeded.
    rng = random.Random(20260828)
    ratios = []
    for _ in range(100_000):
        ra = [a[rng.randrange(5)] for _ in range(5)]
        rb = [b[rng.randrange(3)] for _ in range(3)]
        ratios.append(mean(rb) / mean(ra))
    ratios.sort()
    lo, hi = ratios[2500], ratios[97499]
    print(f"\nbootstrap 95% CI for ratio of means (hit-only): "
          f"[{lo:.2f}, {hi:.2f}]  (point {mean(b)/mean(a):.2f})")
    print(f"fraction of bootstrap ratios <= 1 (no effect): "
          f"{sum(1 for r in ratios if r <= 1.0)/len(ratios):.4f}")
    return {"ratio_point": mean(b) / mean(a), "ratio_ci": (lo, hi),
            "perm_p_8": count / total, "perm_p_9_censored": count9 / tot9}


# ============================== B: leakage ablation ===================

def part_b():
    print("\n" + "=" * 70)
    print("B. leakage-ablation counts")
    print("=" * 70)
    # Reproduce the per-proposal records with bug_class_theory's own code.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks"))
    import sqlite3
    import bug_class_theory as B

    con = sqlite3.connect(f"file:{B.DB}?mode=ro", uri=True)
    rows = list(con.execute(
        """SELECT v.proposal_id, p.operator, v.is_hit, p.proposal_json,
                  v.verdict_json
           FROM verdicts v JOIN proposals p ON p.proposal_id = v.proposal_id"""))
    con.close()
    recs = []
    for pid, op, hit, pj, vj in rows:
        p = json.loads(pj); v = json.loads(vj)
        sim = B.simulate(op, p["tensors"], random.Random(B.SEED))
        if sim is None:
            continue
        rng2 = random.Random(B.SEED)
        spread = 0.0
        for d in p["tensors"].values():
            bf = B.build_flat(d, rng2)
            if bf:
                spread = max(spread, max(bf[0]) - min(bf[0]))
        recs.append(dict(
            op=op, hit=bool(hit), ref_ok=bool(v.get("reference_passed")),
            ref_derived=spread < B.RANGE_LIMIT,
            pred=bool(v.get("reference_passed")) and any(sim.values()),
            pred_off=(spread < B.RANGE_LIMIT) and any(sim.values())))
    n = len(recs)
    acc_rec = sum(1 for r in recs if r["hit"] == r["pred"])
    acc_off = sum(1 for r in recs if r["hit"] == r["pred_off"])
    agree = sum(1 for r in recs if r["ref_ok"] == r["ref_derived"])
    print(f"n={n}; recorded-validity accuracy {acc_rec}/{n}; "
          f"offline accuracy {acc_off}/{n}; term agreement {agree}/{n}")
    assert (n, acc_rec, acc_off, agree) == (120, 120, 112, 83), \
        "banked numbers not reproduced — investigate before trusting CIs"

    w1 = wilson(acc_off, n)
    w2 = wilson(agree, n)
    print(f"\nWilson 95% CIs: offline accuracy {acc_off/n:.3f} "
          f"[{w1[0]:.3f}, {w1[1]:.3f}];  term agreement {agree/n:.3f} "
          f"[{w2[0]:.3f}, {w2[1]:.3f}]")

    # Paired comparison (same 120 items): McNemar exact on discordant pairs.
    b01 = sum(1 for r in recs if r["hit"] == r["pred"] and r["hit"] != r["pred_off"])
    b10 = sum(1 for r in recs if r["hit"] != r["pred"] and r["hit"] == r["pred_off"])
    m = b01 + b10
    # exact binomial two-sided
    p2 = sum(math.comb(m, k) for k in range(0, min(b01, b10) + 1)) * 2 / 2 ** m
    if b01 == m or b10 == m:
        p2 = 2 / 2 ** m  # all discordants one direction
    print(f"paired discordants: recorded-right/offline-wrong {b01}, "
          f"reverse {b10}; McNemar exact two-sided p = {p2:.4g}")

    # Term-level paired structure: where does the 37-case disagreement hide?
    per_op = {}
    for r in recs:
        d = per_op.setdefault(r["op"], [0, 0, 0, 0])  # n, term_disagree, off_err, hit
        d[0] += 1
        d[1] += (r["ref_ok"] != r["ref_derived"])
        d[2] += (r["hit"] != r["pred_off"])
        d[3] += r["hit"]
    print(f"\nper-operator (n, term-disagreements, offline-errors, hits):")
    for op, (nn, td, oe, hh) in sorted(per_op.items()):
        print(f"  {op:14s} n={nn:3d} term_disagree={td:2d} offline_err={oe} hits={hh}")
    sm = per_op.get("softmax", [0, 0, 0, 0])
    n_ns = n - sm[0]
    agree_ns = agree - (sm[0] - sm[1])
    print(f"\ntransfer split: softmax (fitted on) agreement "
          f"{(sm[0]-sm[1])}/{sm[0]} = {(sm[0]-sm[1])/sm[0]:.3f}; "
          f"non-softmax {agree_ns}/{n_ns} = {agree_ns/n_ns:.3f} "
          f"Wilson [{wilson(agree_ns, n_ns)[0]:.3f}, {wilson(agree_ns, n_ns)[1]:.3f}]")

    # Cluster (operator-level) bootstrap for offline accuracy: resample the
    # 6 operators with replacement, pool their proposals.
    ops = sorted(per_op)
    by_op = {op: [r for r in recs if r["op"] == op] for op in ops}
    rng = random.Random(20260828)
    accs = []
    for _ in range(100_000):
        pool = []
        for _ in ops:
            pool.extend(by_op[ops[rng.randrange(len(ops))]])
        accs.append(sum(1 for r in pool if r["hit"] == r["pred_off"]) / len(pool))
    accs.sort()
    print(f"cluster bootstrap 95% CI for offline accuracy: "
          f"[{accs[2500]:.3f}, {accs[97499]:.3f}] (iid Wilson was "
          f"[{w1[0]:.3f}, {w1[1]:.3f}])")
    return {"wilson_off": w1, "wilson_agree": w2, "mcnemar_p": p2,
            "cluster_ci": (accs[2500], accs[97499]),
            "nonsoftmax_agree": (agree_ns, n_ns)}


if __name__ == "__main__":
    ra = part_a()
    rb = part_b()
    out = os.path.join(os.path.dirname(__file__), "..", "data", "uncertainty_results.json")
    with open(out, "w") as f:
        json.dump({"context_effect": {k: v for k, v in ra.items()},
                   "leakage": {k: v for k, v in rb.items()}}, f, indent=1, default=list)
