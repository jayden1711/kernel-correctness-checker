"""
Predictive formula for the n_samples dependence, tested on the 809 banked
nonzero sensitivity vectors.

STEP 1 (exact, distribution-free).  torch.quantile(.,0.95) with linear
interpolation reads index h = 0.95(n-1).  So q95_n is a blend of order
statistics X_(j:n) and X_(j+1:n) with j = floor(h)+1, w = h - floor(h):

        q95_n = (1-w) X_(j:n) + w X_(j+1:n)

For a continuous parent, E[F(X_(j:n))] = j/(n+1).  So the EFFECTIVE parent
quantile the estimator actually targets is

        p(n) = [(1-w) j + w (j+1)] / (n+1) = (h+1)/(n+1) = (0.95n + 0.05)/(n+1)

STEP 2 (model).  Under linearisation s = sigma ||J g||_inf = max of m
correlated |gaussians| -> Gumbel-type upper tail, Q(p) = a + b*G(p),
G(p) = -ln(-ln p).  Then

        tol_n / tol_40 = [a + b G(p(n))] / [a + b G(p(40))]

with the single shape parameter rho = b/a recovered from the sample CV:
CV = (pi/sqrt6) rho / (1 + gamma rho).

Predicts the whole n-curve from ONE number per invocation.  Tested below
against the exact prefix-derived curve.
"""
import gzip, json, math, statistics as st

D = 'verification_runs/n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz'
d = json.load(gzip.open(D))

GAMMA = 0.5772156649
PI_SQ6 = math.pi / math.sqrt(6)


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


def p_eff(n, q=0.95):
    return (q * (n - 1) + 1) / (n + 1)


def G(p):
    return -math.log(-math.log(p))


vecs = []
for e in d['entries']:
    for r in walk(e['mutant']):
        vecs.append(r['sensitivities'])
    for ref in (e.get('refs') or []):
        for r in walk(ref):
            vecs.append(r['sensitivities'])
vecs = [v for v in vecs if max(v) > 0 and min(v) > 0]
print("usable sensitivity vectors (all 40 entries > 0): %d\n" % len(vecs))

print("STEP 1 -- which order statistic q95 actually is, and what parent")
print("quantile it targets in expectation (exact, no distributional model):")
print("%4s  %-26s %10s" % ("n", "q95_n is", "p_eff(n)"))
for n in (2, 3, 5, 10, 15, 20, 21, 22, 30, 40, 100, 1000):
    h = 0.95 * (n - 1); j = math.floor(h) + 1; w = h - math.floor(h)
    desc = "%.2f*X_(%d:%d)+%.2f*X_(%d:%d)" % (1 - w, j, n, w, min(j + 1, n), n)
    if n >= 100:
        desc = "X_(%d:%d) blend" % (j, n)
    print("%4d  %-26s %10.4f" % (n, desc, p_eff(n)))
print("\n  -> at the shipped n=20, `q95` is 0.95*(2nd largest of 20) +")
print("     0.05*(largest of 20), and targets the 90.5th parent percentile,")
print("     not the 95th.  It reaches 0.95 only as n -> infinity.\n")

NS = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40]
print("STEP 2 -- predicted vs measured tolerance ratio  tol_n / tol_40")
print("%4s %10s %10s %10s %10s" % ("n", "measured", "predicted", "rel err", "p_eff"))
G40 = G(p_eff(40))
tot = {n: [] for n in NS}
pred = {n: [] for n in NS}
for v in vecs:
    mean = st.fmean(v); sd = st.stdev(v)
    cv = sd / mean
    # invert CV = PI_SQ6*rho/(1+GAMMA*rho)  for rho = b/a
    denom = (PI_SQ6 - GAMMA * cv)
    if denom <= 0:
        continue
    rho = cv / denom
    q40 = qlin(sorted(v), 0.95)
    if q40 <= 0:
        continue
    for n in NS:
        if n < 2:
            m_ = min(v[:n])
            tot[n].append(m_ / q40)
            pred[n].append(float('nan'))
            continue
        tot[n].append(qlin(sorted(v[:n]), 0.95) / q40)
        pred[n].append((1 + rho * G(p_eff(n))) / (1 + rho * G40))

for n in NS:
    meas = st.median(tot[n])
    pv = [p for p in pred[n] if p == p]
    if not pv:
        print("%4d %10.4f %10s %10s %10.4f" % (n, meas, "-", "-", p_eff(n)))
        continue
    pr = st.median(pv)
    print("%4d %10.4f %10.4f %9.1f%% %10.4f"
          % (n, meas, pr, 100 * (pr - meas) / meas, p_eff(n)))

print()
print("per-invocation accuracy of the formula (relative error of tol_n/tol_40):")
print("%4s %10s %10s %10s" % ("n", "med |err|", "q90 |err|", "max |err|"))
for n in NS:
    if n < 2:
        continue
    errs = sorted(abs(p - m) / m for p, m in zip(pred[n], tot[n])
                  if p == p and m > 0)
    if not errs:
        continue
    print("%4d %9.1f%% %9.1f%% %9.1f%%"
          % (n, 100 * errs[len(errs) // 2], 100 * errs[int(.9 * len(errs))],
             100 * errs[-1]))
