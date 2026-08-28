"""Inspect the three operators where replay CPU/GPU disagreed, plus account
for the 240-222 = 18 missing primary records."""
import gzip, json, math, statistics as st
from collections import Counter

d = json.load(gzip.open(
    'verification_runs/n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz'))


def rec_named(records, name):
    for r in records or []:
        if r.get("name") == name:
            return r
    return None


def primary_sens(records):
    r = rec_named(records, "perturbation_tolerance")
    if not r:
        return None, None
    for sc in (r.get("subchecks") or []):
        if isinstance(sc, dict) and sc.get("kind") == "perturbation_sensitivities":
            return sc, r
    return None, r


print("=" * 84)
print("A. Accounting for the missing primary perturbation records")
print("=" * 84)
missing = Counter(); outcomes = Counter()
for i, e in enumerate(d['entries']):
    invs = [e['mutant']['records']] + [r['records'] for r in (e.get('refs') or [])]
    for recs in invs:
        sc, r = primary_sens(recs)
        if sc is None:
            missing[e['op']] += 1
            outcomes[(e['op'], (r or {}).get('outcome'), ((r or {}).get('detail') or '')[:60])] += 1
print("missing primary sensitivity records: %d" % sum(missing.values()))
for k, v in outcomes.most_common():
    print("   %-28s outcome=%-6s n=%d\n      detail: %s" % (k[0], k[1], v, k[2]))

print()
print("=" * 84)
print("B. cross_entropy -- the 13x outlier")
print("=" * 84)
for i, e in enumerate(d['entries']):
    if e['op'] != 'cross_entropy':
        continue
    invs = [('mutant', e['mutant']['records'])] + \
           [('ref%d' % j, r['records']) for j, r in enumerate(e.get('refs') or [])]
    for tag, recs in invs:
        sc, r = primary_sens(recs)
        if not sc:
            continue
        s = sc['sensitivities']
        print("  %-6s n=%d  min %.4e  med %.4e  max %.4e  n_zero=%d"
              % (tag, len(s), min(s), st.median(s), max(s), sum(1 for v in s if v == 0)))
        print("         first 8: %s" % ["%.3e" % v for v in s[:8]])
        print("         adaptive_tol=%.4e  max_err=%.4e" % (sc['adaptive_tol'], sc['max_err']))
    # what other perturbation-bearing records exist for this op?
    names = [r.get('name') for r in e['mutant']['records'] or []]
    print("  check records present on the mutant run: %s" % names)
    break

print()
print("=" * 84)
print("C. argmin / argmax -- discrete outputs")
print("=" * 84)
for op in ('argmax', 'argmin'):
    for i, e in enumerate(d['entries']):
        if e['op'] != op:
            continue
        invs = [('mutant', e['mutant']['records'])] + \
               [('ref%d' % j, r['records']) for j, r in enumerate(e.get('refs') or [])]
        for tag, recs in invs[:3]:
            sc, r = primary_sens(recs)
            if not sc:
                continue
            s = sc['sensitivities']
            print("  %-8s %-6s distinct values in the 40-vector: %s"
                  % (op, tag, sorted(set(s))[:10]))
        break

print()
print("=" * 84)
print("D. scaled_dot_product_attention vs flash_attention (same math)")
print("=" * 84)
for op in ('flash_attention', 'scaled_dot_product_attention', 'causal_flash_attention'):
    vals = []
    for i, e in enumerate(d['entries']):
        if e['op'] != op:
            continue
        invs = [e['mutant']['records']] + [r['records'] for r in (e.get('refs') or [])]
        for recs in invs:
            sc, _ = primary_sens(recs)
            if sc:
                vals.append(3 * sorted(sc['sensitivities'])[38])
    if vals:
        print("  %-30s n=%2d  median 3*X_(39:40) = %.4e   spread %.4e .. %.4e"
              % (op, len(vals), st.median(vals), min(vals), max(vals)))
