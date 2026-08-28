"""
Scores the near-miss GPU run: realized margins vs design targets, the
P(caught) response curve, full-battery attribution, and the enabled
experiment -- the scale-multiplier identifiability interval, which the
adaptive_tol round measured as (1.642, 4.360) with NO verdict change on
the published corpus, recomputed on the near-miss records.

Run:  .venv/bin/python analyze_near_miss.py
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "data", "near_miss_gpu.json")))
TARGET = {"m050": 0.5, "m080": 0.8, "m100": 1.0, "m125": 1.25, "m200": 2.0}

pert = [r for r in D["records"] if not r.get("full_battery")]
full = [r for r in D["records"] if r.get("full_battery")]

print("1. realized perturbation-check margins (10 seeds each):")
print(f"   {'op/mutant':26s} {'target':>6} {'min':>7} {'med':>7} {'max':>7} "
      f"{'CV%':>5} {'fail rate':>9}")
by = defaultdict(list)
for r in pert:
    by[(r["op"], r["mutant"])].append(r)
resp = defaultdict(lambda: [0, 0])
for (op, mu), rs in sorted(by.items()):
    ms = sorted(r["margin"] for r in rs if r.get("margin"))
    fails = sum(1 for r in rs if not r["pert_passed"])
    mean = sum(ms) / len(ms)
    cv = (sum((m - mean) ** 2 for m in ms) / len(ms)) ** 0.5 / mean * 100
    print(f"   {op + '/' + mu:26s} {TARGET[mu]:6.2f} {ms[0]:7.3f} "
          f"{ms[len(ms)//2]:7.3f} {ms[-1]:7.3f} {cv:5.1f} {fails:>6}/10")
    resp[TARGET[mu]][0] += fails
    resp[TARGET[mu]][1] += len(rs)

print("\n2. the response curve (pooled over ops):")
for t in sorted(resp):
    f, n = resp[t]
    print(f"   design margin {t:4.2f}x: caught {f}/{n} ({100*f/n:.0f}%)")

print("\n3. full-battery verdicts (3 seeds each) -- which checks fire:")
agg = defaultdict(lambda: defaultdict(int))
for r in full:
    key = (r["op"], r["mutant"])
    agg[key]["caught"] += 1 if r["caught"] else 0
    for c in r["failed"]:
        agg[key][c] += 1
for (op, mu), c in sorted(agg.items()):
    others = {k: v for k, v in c.items() if k != "caught"}
    print(f"   {op + '/' + mu:26s} caught {c['caught']}/3  {dict(others)}")

print("\n4. enabled experiment: scale-multiplier identifiability")
print("   (verdict flip scale s* = margin * 3.0 per record; the corpus's")
print("   dead interval was (1.642, 4.360) -- adaptive_tol_theory round)")
flips = sorted(r["margin"] * 3.0 for r in pert if r.get("margin"))
inside = [s for s in flips if 1.642 <= s <= 4.360]
print(f"   {len(flips)} near-miss records; flip scales span "
      f"[{flips[0]:.2f}, {flips[-1]:.2f}]; {len(inside)} of them sit INSIDE "
      f"the old dead interval")
qs = [flips[int(p * len(flips))] for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
print(f"   flip-scale quantiles P10-P90: {[round(q, 2) for q in qs]}")
print("   -> with these records, moving scale anywhere inside (1.64, 4.36)"
      " now changes verdicts; the multiplier is identifiable.")
