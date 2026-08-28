"""
Extreme-value model of the delegation-guard contention ratio (item 6).

Data: the banked per-execution ratios from the 1d investigation --
PRE-fix (sequential two-block construction, forkserver_2026-08-21/
race_rate.jsonl, ~2765 reference-vs-itself executions under 4-way
contention) and POST-fix (interleaved best-of-5 min construction,
race_rate_POSTFIX.jsonl).

Model. Pre-fix, ratio = (T0 + S)/T0' where a scheduling stall S lands in
one block: the ratio tail inherits the stall tail. Test whether the tail
is regularly varying (Frechet domain, max ~ n^{1/alpha}) via (a) the Hill
estimator over the top order statistics and (b) the banked two-point max
growth 23.3@560 -> 51.2@2765. Post-fix, a fire needs EVERY round's ref
time inflated relative to the interleaved candidate time, so the tail
should collapse to ~q(t)^5; measured directly.

Output: the FP surface -- P(ratio >= t) and expected max in N executions
vs threshold t, pre- and post-fix, i.e. exactly "where do FPs appear if
the 10x threshold is tightened".

Run:  .venv/bin/python tail_model.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FS = os.path.join(HERE, "..", "..", "forkserver_2026-08-21")


def load(name):
    out = []
    with open(os.path.join(FS, name)) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("ratio")
            if v is not None and r.get("reached", True):
                out.append(float(v))
    return out


def quant(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def hill(xs, k):
    """Hill tail-index estimator from the top-k order statistics."""
    xs = sorted(xs, reverse=True)[: k + 1]
    logs = [math.log(x / xs[k]) for x in xs[:k]]
    return k / sum(logs)


def main():
    pre = load("race_rate.jsonl")
    post = load("race_rate_POSTFIX.jsonl")
    print(f"pre-fix n={len(pre)}  p50={quant(pre,.5):.2f} p90={quant(pre,.9):.2f} "
          f"p99={quant(pre,.99):.2f} max={max(pre):.2f}")
    print(f"post-fix n={len(post)} p50={quant(post,.5):.2f} p90={quant(post,.9):.2f} "
          f"p99={quant(post,.99):.2f} max={max(post):.2f}")

    # --- tail index, pre-fix -------------------------------------------
    print("\nHill tail-index (pre-fix), k = top order statistics used:")
    for k in (25, 50, 100, 200):
        if k < len(pre):
            print(f"  k={k:4d}: alpha = {hill(pre, k):.2f}")
    # two-point max-growth check (banked: 23.26 at n=560, 51.24 at n=2765)
    a_growth = math.log(2765 / 560) / math.log(51.24 / 23.26)
    print(f"  banked max-growth two-point estimate: alpha = {a_growth:.2f} "
          f"(Frechet max ~ n^(1/alpha))")

    # --- FP surface -----------------------------------------------------
    print("\nFP surface: empirical P(ratio >= t) and expected fires per 10k "
          "executions")
    print(f"  {'t':>6} | {'pre-fix P':>10} {'per-10k':>8} | "
          f"{'post-fix P':>10} {'per-10k':>8}")
    for t in (1.15, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0):
        p_pre = sum(1 for x in pre if x >= t) / len(pre)
        p_post = sum(1 for x in post if x >= t) / len(post)
        print(f"  {t:6.2f} | {p_pre:10.4f} {p_pre*1e4:8.1f} | "
              f"{p_post:10.4f} {p_post*1e4:8.1f}")

    # --- expected max in N, both constructions --------------------------
    # pre: Frechet extrapolation from fitted alpha at p99 anchor
    alpha = hill(pre, 100) if len(pre) > 100 else 2.0
    x99 = quant(pre, .99)
    print(f"\npre-fix Frechet extrapolation (alpha={alpha:.2f}, anchored at "
          f"p99={x99:.2f}): expected max ~ x99 * (0.01 N)^(1/alpha)")
    for N in (560, 2765, 10000, 100000):
        print(f"  N={N:6d}: predicted max ~ {x99 * (0.01*N)**(1/alpha):8.1f}")
    print(f"  (measured: 23.26 @ 560, 51.24 @ 2765)")

    # post: is the post-fix tail even extrapolatable? report largest few
    tail = sorted(post, reverse=True)[:8]
    print(f"\npost-fix top ratios: {[round(x, 3) for x in tail]}")
    # min-of-5 collapse: per-round exceedance q(t) bounds the min-based
    # rate by ~q^5 under round independence
    for t, q in ((1.1, None), (1.2, None)):
        qe = sum(1 for x in pre if x >= t) / len(pre)   # proxy per-round
        print(f"  independence bound at t={t}: per-round q~{qe:.3f} -> "
              f"min-of-5 ~ q^5 = {qe**5:.2e} per execution")


if __name__ == "__main__":
    main()
