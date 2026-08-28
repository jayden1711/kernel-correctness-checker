"""Which INPUTS produce the linear-regime violations in the attention family?
Attribute every banked sensitivity vector to the check record that produced it
(perturbation_tolerance = primary input; adversarial_<name> = a spec-declared
adversarial input)."""
import gzip, json, math, statistics as st
from collections import defaultdict

CEIL = math.sqrt(math.pi / 2 - 1)
d = json.load(gzip.open(
    'verification_runs/n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz'))

rows = []
for e in d['entries']:
    op = e['op']
    invs = [('mutant:' + e['mutant']['name'], e['mutant']['records'])] + \
           [('ref%d' % j, r['records']) for j, r in enumerate(e.get('refs') or [])]
    for tag, recs in invs:
        for r in (recs or []):
            for sc in (r.get('subchecks') or []):
                if isinstance(sc, dict) and sc.get('kind') == 'perturbation_sensitivities':
                    rows.append((op, r.get('name'), tag, sc['sensitivities'],
                                 sc['adaptive_tol'], sc['max_err']))

print("=" * 96)
print("ATTENTION FAMILY -- CV by which INPUT the check ran on")
print("=" * 96)
print("%-28s %-34s %5s %8s %8s %6s" %
      ("op", "input (check record name)", "n", "CV med", "CV max", "viol"))
agg = defaultdict(list)
for op, cname, tag, s, tol, me in rows:
    if 'attention' not in op:
        continue
    if st.fmean(s) <= 0:
        continue
    agg[(op, cname)].append(st.stdev(s) / st.fmean(s))
for (op, cname), cvs in sorted(agg.items()):
    v = sum(1 for c in cvs if c > 1.0)
    print("%-28s %-34s %5d %8.4f %8.4f %6s"
          % (op, cname, len(cvs), st.median(cvs), max(cvs), v if v else ""))

print()
print("=" * 96)
print("ALL OPERATORS -- every check-record name that produced a CV > 1.0")
print("=" * 96)
bad = defaultdict(int); tot = defaultdict(int)
for op, cname, tag, s, tol, me in rows:
    if st.fmean(s) <= 0:
        continue
    tot[(op, cname)] += 1
    if st.stdev(s) / st.fmean(s) > 1.0:
        bad[(op, cname)] += 1
for k in sorted(bad, key=lambda k: -bad[k]):
    print("   %-28s %-34s %d / %d invocations" % (k[0], k[1], bad[k], tot[k]))

print()
print("=" * 96)
print("PRIMARY INPUT ONLY (perturbation_tolerance record) -- the configuration")
print("the coverage table measures")
print("=" * 96)
agg2 = defaultdict(list)
for op, cname, tag, s, tol, me in rows:
    if cname != 'perturbation_tolerance' or st.fmean(s) <= 0:
        continue
    agg2[op].append(st.stdev(s) / st.fmean(s))
nviol = 0
for op in sorted(agg2):
    cvs = agg2[op]
    v = sum(1 for c in cvs if c > 1.0)
    nviol += v
    flag = "  <-- VIOLATION" if v else ""
    print("   %-30s n=%3d  CV med %.4f  max %.4f%s"
          % (op, len(cvs), st.median(cvs), max(cvs), flag))
print("\n   total primary-input invocations exceeding the ceiling: %d" % nviol)

# adversarial input names available for the attention specs
print()
print("spec-declared adversarial inputs for the attention family:")
import subprocess
for f in ('flash_attention', 'causal_flash_attention', 'scaled_dot_product_attention'):
    try:
        src = open('verification/specs/%s.py' % f).read()
        names = [ln.strip() for ln in src.splitlines() if 'return [' in ln or '("' in ln]
        picks = [n for n in names if n.startswith('("')]
        print("   %-30s %s" % (f, [p.split('"')[1] for p in picks]))
    except Exception as ex:
        print("   %-30s (%s)" % (f, ex))
