"""Final tables for both parts."""
import json, math, statistics as st
from collections import defaultdict

HERE = __file__.rsplit('/', 1)[0]
G = [json.loads(l) for l in open(HERE + '/gen_native.jsonl')]
N = [json.loads(l) for l in open(HERE + '/gpu_native.jsonl')]
N = [r for r in N if r.get('kind') == 'primary' and 'error' not in r]

meas_y = defaultdict(list)
for r in N:
    meas_y[r['op']].append(r['tol'] / (3 * r['sigma'] * r['L']))

byop = defaultdict(list)
for r in G:
    byop[r['op']].append(r)

print("=" * 122)
print("PART A -- STRUCTURAL L vs PROBED L, both on the real Triton kernels")
print("=" * 122)
print("%-30s %7s %6s %11s %9s %9s %9s %7s" %
      ("operator", "kind", "m", "L_struct", "mc400/st", "mc4k/st", "mc20k/st", "spread"))
r400, r4k, r20k = [], [], []
for op in sorted(byop):
    v = byop[op]
    kind = "STATIC" if v[0]['static'] else "closed"
    a = st.median([r['L_mc']['400'] / r['L_struct'] for r in v])
    b = st.median([r['L_mc']['4000'] / r['L_struct'] for r in v])
    c = st.median([r['L_mc']['20000'] / r['L_struct'] for r in v])
    r400.append(a); r4k.append(b); r20k.append(c)
    print("%-30s %7s %6d %11.4e %9.3f %9.3f %9.3f %7.1f"
          % (op, kind, v[0]['m'], st.median([r['L_struct'] for r in v]), a, b, c,
             st.median([r['spread'] for r in v])))
print("-" * 122)
for nm, rr in (("K=400", r400), ("K=4000", r4k), ("K=20000", r20k)):
    print("  L_mc/L_struct at %-8s min %.3f  median %.3f  max %.3f   (within +/-5%%: %d/%d)"
          % (nm, min(rr), st.median(rr), max(rr),
             sum(1 for x in rr if 0.95 <= x <= 1.05), len(rr)))

# predicted estimator bias: max over m of a chi-K/K rms estimate
print()
print("  predicted K=400 bias from the estimator alone, 1 + sqrt(2 ln m)/sqrt(2K):")
for op in sorted(byop)[:0]:
    pass
pb = []
for op in sorted(byop):
    v = byop[op]
    m = v[0]['n_rows']
    pred = 1 + math.sqrt(2 * math.log(max(m, 2))) / math.sqrt(2 * 400)
    act = st.median([r['L_mc']['400'] / r['L_struct'] for r in v])
    pb.append((act, pred, op))
print("     median predicted %.3f vs median actual %.3f"
      % (st.median([p for _, p, _ in pb]), st.median([a for a, _, _ in pb])))

print()
print("=" * 122)
print("PART B -- can y = tol/(3 sigma L) be predicted?")
print("=" * 122)
print("%-30s %6s %8s %9s %9s %9s %9s" %
      ("operator", "m", "y meas", "M1 lead", "M1/meas", "M3 rows", "M3/meas"))
m1r, m3r = [], []
for op in sorted(byop):
    v = byop[op]
    ym = st.median(meas_y[op])
    lead = 0.7537 * math.sqrt(2 * math.log(2 * v[0]['m']))
    m3 = st.median([r['y_M3'] for r in v])
    m1r.append(lead / ym); m3r.append(m3 / ym)
    print("%-30s %6d %8.3f %9.3f %9.3f %9.3f %9.3f"
          % (op, v[0]['m'], ym, lead, lead / ym, m3, m3 / ym))
print("-" * 122)
for nm, rr in (("M1  (sigma,L,m only)", m1r), ("M3  (row-norm profile)", m3r)):
    print("  %-24s pred/meas  min %.3f  median %.3f  max %.3f  SPREAD %.2fx  within +/-10%%: %d/%d"
          % (nm, min(rr), st.median(rr), max(rr), max(rr) / min(rr),
             sum(1 for x in rr if 0.9 <= x <= 1.1), len(rr)))


def r2(pred, act):
    mu = sum(act) / len(act)
    ss = sum((a - mu) ** 2 for a in act)
    return 1 - sum((p - a) ** 2 for p, a in zip(pred, act)) / ss


acts = [st.median(meas_y[op]) for op in sorted(byop)]
p1 = [0.7537 * math.sqrt(2 * math.log(2 * byop[op][0]['m'])) for op in sorted(byop)]
p3 = [st.median([r['y_M3'] for r in byop[op]]) for op in sorted(byop)]
print()
print("  R^2 on the 27 per-operator medians:   M1 = %.4f     M3 = %.4f"
      % (r2(p1, acts), r2(p3, acts)))
print()
print("  worst M3 over-predictions (correlated rows):")
tab = sorted(zip(m3r, sorted(byop)), reverse=True)
for x, op in tab[:6]:
    print("     %-30s M3/meas = %.3f" % (op, x))
print("  worst M3 under-predictions:")
for x, op in tab[-4:]:
    print("     %-30s M3/meas = %.3f" % (op, x))
