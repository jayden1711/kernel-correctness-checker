"""
Is adaptive_tol predictable from (sigma, L, m, n) with ONE set of constants?

The theorem gives  tol = 3 sigma * q95_n( max_i |<J_i/L, g>| ) * L, so the
dimensionless shape factor

        y = tol / (3 sigma L)

should be a function of m and n alone.  n is fixed at 40 here (its dependence
is separately established and validated to <=4.3% in FINDINGS.md 5), so this
asks whether y = h(m) with one h across all 27 operators.

Candidate models, fitted by least squares on the 228 native GPU invocations:
    M0  y = c                       (null: no m dependence)
    M1  y = a sqrt(2 ln 2m) + b     (the theorem's leading term)
    M2  y = a (ln m)^c + b          (free exponent)
    M3  y = a sqrt(2 ln 2 m_eff)    with m_eff from the row-norm profile
                                    (fitted separately in fit_meff.py)
"""
import json, math, statistics as st

HERE = __file__.rsplit('/', 1)[0]
rows = [json.loads(l) for l in open(HERE + '/gpu_native.jsonl')]
P = [r for r in rows if r.get('kind') == 'primary' and 'error' not in r]
print("native invocations: %d   operators: %d" % (len(P), len(set(r['op'] for r in P))))

for r in P:
    r['y'] = r['tol'] / (3 * r['sigma'] * r['L'])
    r['lead'] = math.sqrt(2 * math.log(2 * r['m']))


def lstsq(X, yv):
    """Tiny normal-equations solver; X is a list of rows (lists)."""
    p = len(X[0])
    A = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(p)]
         for i in range(p)]
    b = [sum(X[k][i] * yv[k] for k in range(len(X))) for i in range(p)]
    # gaussian elimination
    M = [A[i][:] + [b[i]] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r_: abs(M[r_][c]))
        M[c], M[piv] = M[piv], M[c]
        for r_ in range(p):
            if r_ == c or M[c][c] == 0:
                continue
            f = M[r_][c] / M[c][c]
            for k in range(c, p + 1):
                M[r_][k] -= f * M[c][k]
    return [M[i][p] / M[i][i] for i in range(p)]


def r2(pred, act):
    mu = sum(act) / len(act)
    ss = sum((a - mu) ** 2 for a in act)
    rs = sum((p - a) ** 2 for p, a in zip(pred, act))
    return 1 - rs / ss if ss else float('nan')


yv = [r['y'] for r in P]
print("\ny = tol/(3 sigma L):  min %.3f  median %.3f  max %.3f  (spread %.2fx)"
      % (min(yv), st.median(yv), max(yv), max(yv) / min(yv)))

# ---- M0 null
c0 = sum(yv) / len(yv)
print("\nM0  y = c                     c = %.3f              R2 = %.4f"
      % (c0, r2([c0] * len(yv), yv)))

# ---- M1 leading term
X = [[r['lead'], 1.0] for r in P]
a, b = lstsq(X, yv)
p1 = [a * r['lead'] + b for r in P]
print("M1  y = a*sqrt(2 ln 2m) + b   a = %.4f  b = %.4f   R2 = %.4f"
      % (a, b, r2(p1, yv)))

# ---- M1' no intercept (the theorem's actual shape)
num = sum(r['lead'] * r['y'] for r in P); den = sum(r['lead'] ** 2 for r in P)
a1 = num / den
p1b = [a1 * r['lead'] for r in P]
print("M1' y = a*sqrt(2 ln 2m)       a = %.4f              R2 = %.4f"
      % (a1, r2(p1b, yv)))

# ---- M2 free exponent, grid over c
best = None
for cexp in [i / 100 for i in range(5, 205)]:
    X = [[math.log(max(r['m'], 1.0001)) ** cexp, 1.0] for r in P]
    try:
        aa, bb = lstsq(X, yv)
    except ZeroDivisionError:
        continue
    pp = [aa * math.log(max(r['m'], 1.0001)) ** cexp + bb for r in P]
    sc = r2(pp, yv)
    if best is None or sc > best[0]:
        best = (sc, cexp, aa, bb)
print("M2  y = a*(ln m)^c + b        a = %.4f  b = %.4f  c = %.2f   R2 = %.4f"
      % (best[2], best[3], best[1], best[0]))

# ---- residuals by operator, for the theorem's own model M1'
print()
print("=" * 96)
print("PER-OPERATOR RESIDUALS under the theorem's leading term  y = a*sqrt(2 ln 2m)")
print("(a = %.4f fitted across all operators)" % a1)
print("=" * 96)
print("%-30s %5s %8s %8s %8s %9s" % ("operator", "m", "y meas", "y pred", "ratio", "resid %"))
byop = {}
for r in P:
    byop.setdefault(r['op'], []).append(r)
tab = []
for op in sorted(byop):
    v = byop[op]
    ym = st.median([r['y'] for r in v])
    yp = a1 * v[0]['lead']
    tab.append((ym / yp, op, v[0]['m'], ym, yp))
for ratio, op, m, ym, yp in sorted(tab):
    print("%-30s %5d %8.3f %8.3f %8.3f %+8.1f%%"
          % (op, m, ym, yp, ratio, 100 * (ym - yp) / yp))

rr = [t[0] for t in tab]
print()
print("ratio measured/predicted across the 27 operators:")
print("   min %.3f   median %.3f   max %.3f   SPREAD %.2fx"
      % (min(rr), st.median(rr), max(rr), max(rr) / min(rr)))
within = sum(1 for x in rr if 0.8 <= x <= 1.25)
print("   within +/-25%%: %d of %d operators" % (within, len(rr)))

# ---- how much of the spread is m_eff (correlation) rather than noise? -------
print()
print("The residual is systematic, not sampling noise -- within-operator spread")
print("of y compared with between-operator spread:")
wi = st.median([ (max(r['y'] for r in v) / min(r['y'] for r in v)) for v in byop.values() if len(v) > 1 ])
print("   median within-operator max/min of y : %.3f" % wi)
print("   between-operator max/min of median y: %.3f"
      % (max(t[3] for t in tab) / min(t[3] for t in tab)))
