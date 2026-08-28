"""Exact (n_samples, scale) response surface from banked n=40 sensitivity vectors.

The verdict is  fail <=> max_err > max(scale * q95(sens[:n]), 1e-6).
Both max_err and the full 40-vector are recorded, and the n-sample vector is a
prefix of the 40-sample one, so EVERY (n, scale) verdict is determined exactly
offline. No rerun, no RNG noise.
"""
import gzip, json, math
from collections import defaultdict

D = 'verification_runs/n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz'
d = json.load(gzip.open(D))


def walk(o):
    if isinstance(o, dict):
        if o.get('kind') == 'perturbation_sensitivities':
            yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def qlin(sorted_s, q):
    n = len(sorted_s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return sorted_s[lo] + (h - lo) * (sorted_s[hi] - sorted_s[lo])


# collect (kind, op, mutant_name, sensitivities, max_err); one row per invocation
rows = []
for e in d['entries']:
    op = e['op']
    for r in walk(e['mutant']):
        rows.append(('mutant', op, e['mutant']['name'], r['sensitivities'], r['max_err']))
    for ref in (e.get('refs') or []):
        for r in walk(ref):
            rows.append(('ref', op, '-', r['sensitivities'], r['max_err']))

print("invocations: %d   (mutant %d / ref %d)" %
      (len(rows), sum(1 for r in rows if r[0] == 'mutant'),
       sum(1 for r in rows if r[0] == 'ref')))

zero = sum(1 for r in rows if r[4] == 0.0)
print("max_err exactly 0.0: %d (%.1f%%)  -> cannot flip at ANY (n, scale)"
      % (zero, 100 * zero / len(rows)))
live = [r for r in rows if r[4] > 0.0]
print("max_err > 0 (the only rows that can respond): %d" % len(live))
lm = sum(1 for r in live if r[0] == 'mutant')
print("   of which mutant %d / ref %d" % (lm, len(live) - lm))
print()

# ---- per-invocation critical multiplier c* = max_err / q95(sens[:n]) --------
# invocation FAILS at scale c  <=>  max_err > max(c*q95_n, 1e-6)
def crit(sens, n, max_err):
    """Smallest scale c at which this invocation PASSES (i.e. fails for c<c*).
    Returns +inf if it passes at every c (max_err <= 1e-6 floor)."""
    q = qlin(sorted(sens[:n]), 0.95)
    if max_err <= 1e-6:
        return 0.0        # passes even at c=0 because of the floor
    if q <= 0:
        return float('inf')   # only the floor protects it; needs c=inf
    return max_err / q

NS = [1, 2, 3, 5, 10, 15, 20, 30, 40]
CS = [0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0, 1e4, 1e6]

print("=" * 104)
print("FALSE POSITIVES: reference invocations rejected, out of %d"
      % sum(1 for r in rows if r[0] == 'ref'))
print("=" * 104)
hdr = "%4s |" % "n" + "".join("%9s" % ("c=%g" % c) for c in CS)
print(hdr); print("-" * len(hdr))
for n in NS:
    line = "%4d |" % n
    for c in CS:
        k = 0
        for kind, op, mn, sens, me in rows:
            if kind != 'ref':
                continue
            if me <= 1e-6:
                continue
            q = qlin(sorted(sens[:n]), 0.95)
            if me > max(c * q, 1e-6):
                k += 1
        line += "%9d" % k
    print(line)

print()
print("=" * 104)
print("CATCHES: mutant invocations rejected by THIS check, out of %d"
      % sum(1 for r in rows if r[0] == 'mutant'))
print("=" * 104)
print(hdr); print("-" * len(hdr))
for n in NS:
    line = "%4d |" % n
    for c in CS:
        k = 0
        for kind, op, mn, sens, me in rows:
            if kind != 'mutant':
                continue
            q = qlin(sorted(sens[:n]), 0.95)
            if me > max(c * q, 1e-6):
                k += 1
        line += "%9d" % k
    print(line)

print()
print("=" * 104)
print("MARGIN DISTRIBUTION at the shipped n=20: ratio max_err / (3.0 * q95)")
print("  <1 = passes, >1 = fails.  Distance from 1 is how far `scale` could move.")
print("=" * 104)
for kind in ('ref', 'mutant'):
    rr = []
    for k, op, mn, sens, me in rows:
        if k != kind or me <= 0:
            continue
        q = qlin(sorted(sens[:20]), 0.95)
        tol = max(3.0 * q, 1e-6)
        rr.append((me / tol, op, mn))
    rr.sort()
    print("\n%s: %d invocations with max_err>0" % (kind.upper(), len(rr)))
    if not rr:
        continue
    print("  min      %.4g   (%s/%s)" % (rr[0][0], rr[0][1], rr[0][2]))
    for q in (.25, .5, .75):
        i = int(q * (len(rr) - 1))
        print("  q%02d      %.4g" % (q * 100, rr[i][0]))
    print("  max      %.4g   (%s/%s)" % (rr[-1][0], rr[-1][1], rr[-1][2]))
    below = [r for r in rr if r[0] < 1]
    above = [r for r in rr if r[0] >= 1]
    print("  passes %d / fails %d" % (len(below), len(above)))
    if below:
        print("  closest PASS to the boundary : ratio %.4g  (%s/%s)"
              % (below[-1][0], below[-1][1], below[-1][2]))
    if above:
        print("  closest FAIL to the boundary : ratio %.4g  (%s/%s)"
              % (above[0][0], above[0][1], above[0][2]))
