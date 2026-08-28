"""
Does forkserver change the delegation-detector race rate? Analysis.

Reads `race_rate.jsonl` and answers on TWO endpoints, because either one alone
can mislead:

  BINARY      the flip rate itself (`delegation_ratio > 10`). This is what costs
              a verdict, but it is a rare event, so a null result here means
              little without the power calculation printed alongside it.

  CONTINUOUS  the distribution of `delegation_ratio`. Far more powerful for the
              same data. If the arms are indistinguishable through the bulk AND
              the upper quantiles, the tail rates cannot meaningfully differ.

Reports a confidence interval and a MINIMUM DETECTABLE EFFECT for the binary
test, so that a null is reported as "ruled out down to X" rather than as
"no difference" -- §5 instance 12's rule, which this project has now been
bitten by twice, once in each direction.

Plain python3. No numpy.
"""
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "race_rate.jsonl")
THRESHOLD = 10.0          # runtime_guards.py: t_cand < t_ref * 0.1


def wilson(k, n, z=1.959964):
    """Wilson score interval -- correct at small k, unlike normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pv


def mde(p_base, n_per_arm, alpha=0.05, power=0.80):
    """Smallest p2 detectable at this n, by search."""
    za, zb = 1.959964, 0.8416212
    p2 = p_base
    for _ in range(200000):
        p2 += 0.0001
        if p2 >= 1:
            return None
        pbar = (p_base + p2) / 2
        need = ((za * math.sqrt(2 * pbar * (1 - pbar))
                 + zb * math.sqrt(p_base * (1 - p_base) + p2 * (1 - p2))) ** 2
                / (p2 - p_base) ** 2)
        if need <= n_per_arm:
            return p2
    return None


def quantile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def mannwhitney_z(a, b):
    """Normal-approximation Mann-Whitney U. Non-parametric: the ratio
    distribution is heavy-tailed and a t-test on it would be reading noise."""
    if not a or not b:
        return float("nan"), float("nan")
    merged = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = {}
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    r1 = sum(ranks[k] for k, (_, g) in enumerate(merged) if g == 0)
    n1, n2 = len(a), len(b)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sd == 0:
        return 0.0, 1.0
    z = (u1 - mu) / sd
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pv


def main():
    if not os.path.exists(SRC):
        print("missing", SRC)
        return 1
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    reached = [r for r in rows if r.get("reached")]

    arms = defaultdict(list)
    for r in reached:
        arms[r["arm"]].append(r)

    print("=" * 74)
    print("  forkserver vs spawn: the delegation-detector race")
    print("=" * 74)
    print(f"  trials recorded          {len(rows)}")
    print(f"  reached the detector     {len(reached)}"
          f"   ({len(rows) - len(reached)} out-of-domain, excluded)")
    passes = len({r["pass_idx"] for r in rows})
    print(f"  passes                   {passes}")
    print()

    # Sanity: the arms must be balanced, or the comparison is confounded by
    # exposure rather than by start method.
    print("  %-12s %8s %8s %10s" % ("arm", "trials", "flips", "rate"))
    stat = {}
    for arm in sorted(arms):
        rs = arms[arm]
        n = len(rs)
        k = sum(1 for r in rs if not r["ke_passed"])
        lo, hi = wilson(k, n)
        stat[arm] = (k, n, lo, hi, [r["ratio"] for r in rs])
        print("  %-12s %8d %8d %9.2f%%   95%% CI [%.2f%%, %.2f%%]"
              % (arm, n, k, 100 * k / n, 100 * lo, 100 * hi))
    print()

    if set(stat) != {"spawn", "forkserver"}:
        print("  need both arms"); return 1

    ks, ns, _, _, rs_s = stat["spawn"]
    kf, nf, _, _, rs_f = stat["forkserver"]

    print("-" * 74)
    print("  BINARY ENDPOINT — the flip rate")
    print("-" * 74)
    z, pv = two_prop_z(ks, ns, kf, nf)
    print(f"  spawn      {ks}/{ns} = {100*ks/ns:.2f}%")
    print(f"  forkserver {kf}/{nf} = {100*kf/nf:.2f}%")
    print(f"  two-proportion z = {z:+.3f}   p = {pv:.3f}")
    base = ks / ns if ns else 0.0
    m = mde(max(base, 1e-4), min(ns, nf))
    if m:
        print(f"  MINIMUM DETECTABLE EFFECT at n={min(ns,nf)}/arm, 80% power:")
        print(f"    a rise from {100*base:.2f}% to >= {100*m:.2f}% "
              f"({m/max(base,1e-9):.2f}x) would have been detected.")
        print(f"    Anything smaller is NOT ruled out by this experiment.")
    print()

    print("-" * 74)
    print("  CONTINUOUS ENDPOINT — the delegation_ratio distribution")
    print("-" * 74)
    print("  %-12s %8s %8s %8s %8s %8s %8s"
          % ("arm", "n", "p50", "p90", "p99", "max", ">=10"))
    for arm, rr in (("spawn", rs_s), ("forkserver", rs_f)):
        fin = [v for v in rr if v != float("inf")]
        print("  %-12s %8d %8.2f %8.2f %8.2f %8.2f %8d"
              % (arm, len(rr), quantile(fin, .5), quantile(fin, .9),
                 quantile(fin, .99), max(fin) if fin else float("nan"),
                 sum(1 for v in rr if v >= THRESHOLD)))
    zz, pp = mannwhitney_z(rs_s, rs_f)
    print(f"\n  Mann-Whitney z = {zz:+.3f}   p = {pp:.4f}")
    print("  (non-parametric on purpose: the ratio is heavy-tailed, so a mean "
          "comparison\n   would be dominated by whichever arm happened to draw "
          "the worst outlier)")
    print()

    # Survival curve. The binary endpoint tests ONE threshold, chosen by the
    # production code rather than by this experiment, and it happens to sit
    # where the counts are smallest. Sweeping it shows whether a difference is
    # a coherent shift in the tail or a single noisy cell.
    #
    # These tests are NESTED and therefore heavily correlated -- the smallest p
    # across the sweep must NOT be read as the p-value of anything. It is here
    # to show the SHAPE.
    print("-" * 74)
    print("  SURVIVAL CURVE — the same comparison at every threshold")
    print("-" * 74)
    print("  %8s %10s %12s %9s %8s" % ("thresh", "spawn", "forkserver",
                                       "ratio", "p (uncorr)"))
    for t in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15):
        a = sum(1 for v in rs_s if v >= t)
        b = sum(1 for v in rs_f if v >= t)
        _, pv_t = two_prop_z(a, len(rs_s), b, len(rs_f))
        mark = "  <- production threshold" if t == THRESHOLD else ""
        print("  %8.0f %10d %12d %9s %8.3f%s"
              % (t, a, b, ("%.2fx" % (b / a)) if a else "n/a", pv_t, mark))
    print("  (nested tests: do NOT take the minimum p as a result. Bonferroni "
          "across 12\n   thresholds multiplies any single p by ~12.)")
    print()

    # Order control: the arms were interleaved, so a drift over the run should
    # appear in BOTH. If one arm drifts and the other does not, the comparison
    # is measuring the run, not the start method.
    print("-" * 74)
    print("  ORDER CONTROL — first half vs second half, within each arm")
    print("-" * 74)
    half = passes / 2
    for arm in ("spawn", "forkserver"):
        rs = arms[arm]
        a = [r for r in rs if r["pass_idx"] < half]
        b = [r for r in rs if r["pass_idx"] >= half]
        fa = sum(1 for r in a if not r["ke_passed"])
        fb = sum(1 for r in b if not r["ke_passed"])
        print("  %-12s first half %2d/%3d (%.2f%%)   second half %2d/%3d (%.2f%%)"
              % (arm, fa, len(a), 100*fa/max(len(a),1),
                 fb, len(b), 100*fb/max(len(b),1)))
    print()

    # What the rate means in practice, and what would remove it entirely. The
    # phenomenon being compared is a FALSE POSITIVE of a defective check (item
    # 1d), so the threshold that eliminates it matters more than which arm
    # produces marginally more of them.
    print("-" * 74)
    print("  WHAT THIS COSTS, AND WHAT REMOVES IT")
    print("-" * 74)
    allr = rs_s + rs_f
    worst = max(v for v in allr if v != float("inf"))
    for arm, (k, n, _, _, _) in stat.items():
        print(f"  {arm:11s} {80 * k / n:5.2f} spurious reference failures per "
              f"80-proposal search")
    print(f"\n  Highest ratio seen anywhere in {len(allr)} executions: {worst:.2f}")
    for t in (20, 25, 50):
        print(f"    threshold {t:3d}x -> "
              f"{sum(1 for v in allr if v >= t)} flips across BOTH arms")
    print("  A kernel that genuinely delegates CALLS the reference, so its ratio")
    print("  is ~1.0 and this check never flags it. Raising the threshold "
          "therefore costs")
    print("  no detection power against the thing the message names.")
    print()

    print("=" * 74)
    verdict = ("NO RESOLVABLE DIFFERENCE" if pv >= 0.05 and pp >= 0.05
               else "DIFFERENCE DETECTED")
    print(f"  VERDICT: {verdict}")
    if pv >= 0.05 and pp >= 0.05 and m:
        print(f"  Both endpoints agree. Effects at or above "
              f"{m/max(base,1e-9):.2f}x the spawn rate are ruled out;")
        print(f"  smaller ones are not. Report as a bound, never as 'no effect'.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
