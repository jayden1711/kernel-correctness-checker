"""Where the tolerance actually sits relative to (a) the float32 noise floor
and (b) the 1e-6 absolute floor -- i.e. how much calibration headroom the
mechanism has, and which constant is actually binding."""
import gzip, json, math, statistics as st
from collections import Counter

EPS32 = 2.0 ** -23
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


def qlin(s, q):
    s = sorted(s); n = len(s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


rows = []
for e in d['entries']:
    op = e['op']
    for r in walk(e['mutant']):
        rows.append(('mutant', op, e['mutant']['name'], r))
    for ref in (e.get('refs') or []):
        for r in walk(ref):
            rows.append(('ref', op, '-', r))

print("float32 eps = %.6e\n" % EPS32)

# --- which constant binds: 3*q95 or the 1e-6 floor? -------------------------
print("=" * 78)
print("WHICH CONSTANT BINDS at n=20?  tol = max(3*q95_20, 1e-6)")
print("=" * 78)
binds = Counter(); by_op = {}
for kind, op, mn, r in rows:
    q = qlin(r['sensitivities'][:20], 0.95)
    who = 'floor 1e-6' if 3 * q <= 1e-6 else '3*q95'
    binds[who] += 1
    by_op.setdefault(op, Counter())[who] += 1
tot = sum(binds.values())
for k, v in binds.most_common():
    print("  %-12s %4d / %d  (%.1f%%)" % (k, v, tot, 100 * v / tot))
print("\n  ops where the FLOOR binds on every invocation (J = 0 a.e. --")
print("  discrete / piecewise-constant outputs, exactly as linearisation predicts):")
for op, c in sorted(by_op.items()):
    if c['3*q95'] == 0:
        print("     %-22s %d invocations" % (op, c['floor 1e-6']))
print("  ops where the floor binds on SOME invocations:")
for op, c in sorted(by_op.items()):
    if c['floor 1e-6'] and c['3*q95']:
        print("     %-22s %d floor / %d adaptive" % (op, c['floor 1e-6'], c['3*q95']))

# --- headroom: how far could the band be tightened before the first FP? -----
print()
print("=" * 78)
print("TIGHTENING HEADROOM on reference (correct-kernel) invocations")
print("=" * 78)
ref_nonzero = [(op, r) for kind, op, mn, r in rows if kind == 'ref' and r['max_err'] > 0]
print("reference invocations with max_err > 0: %d of %d"
      % (len(ref_nonzero), sum(1 for k, *_ in rows if k == 'ref')))
for op, r in ref_nonzero:
    q = qlin(r['sensitivities'][:20], 0.95)
    tol = max(3 * q, 1e-6)
    print("  %-16s max_err %.4e = %.2f ulp(f32)   tol %.4e   margin %.1fx"
          % (op, r['max_err'], r['max_err'] / EPS32, tol, tol / r['max_err']))
    print("     3*q95 alone = %.4e (margin %.1fx);  the 1e-6 floor alone = %.1fx"
          % (3 * q, 3 * q / r['max_err'], 1e-6 / r['max_err']))

print()
print("=" * 78)
print("IDENTIFIED INTERVAL for `scale` at n=20 (invocation-level verdicts)")
print("=" * 78)
ratios = []
for kind, op, mn, r in rows:
    if r['max_err'] <= 0:
        continue
    q = qlin(r['sensitivities'][:20], 0.95)
    if q <= 0:
        continue
    ratios.append((r['max_err'] / q, kind, op, mn))   # critical scale c*
ratios.sort()
fails_at_3 = [x for x in ratios if x[0] > 3.0]
pass_at_3 = [x for x in ratios if x[0] <= 3.0]
lo = max((x[0] for x in pass_at_3), default=0.0)
hi = min((x[0] for x in fails_at_3), default=float('inf'))
print("  every invocation with max_err>0 and q95>0: %d" % len(ratios))
print("  nearest PASS below the shipped scale=3.0 : c* = %.4f  (%s %s/%s)"
      % ((pass_at_3[-1] if pass_at_3 else (0, '', '', ''))[0:1] +
         tuple(pass_at_3[-1][1:]) if pass_at_3 else (0, '', '', '')))
print("  nearest FAIL above                        : c* = %.4f  (%s %s/%s)"
      % ((fails_at_3[0][0],) + tuple(fails_at_3[0][1:]) if fails_at_3 else (0, '', '', '')))
print("  => scale can range over (%.3f, %.3f) with NO invocation-level" % (lo, hi))
print("     verdict change anywhere in the corpus.  Shipped value 3.0 sits")
print("     %.2f decades from the lower edge and %.2f from the upper."
      % (math.log10(3.0 / lo) if lo > 0 else float('inf'),
         math.log10(hi / 3.0) if hi < float('inf') else float('inf')))
