"""Checker-wall (sum of per-candidate dt_ms) vs n, from the same banked arms.

`dt_ms` is the whole KernelChecker.run for one candidate, so this is the
denominator the prior rounds quoted their percentages against.
"""
import gzip, json

ARMS = "verification_runs/n_samples_curve_2026-08-25/arms"
NS = [3, 5, 10, 15, 20, 40]

def wall(n):
    d = json.load(gzip.open(f"{ARMS}/VALID_n{n}.json.gz"))
    tot = 0.0
    ncand = 0
    for e in d["entries"]:
        tot += e["mutant"]["dt_ms"]; ncand += 1
        for ref in e.get("refs", []):
            tot += ref["dt_ms"]; ncand += 1
    return tot, ncand

rows = [(n,) + wall(n) for n in NS]
print(f"{'n':>3} {'checker wall ms':>16} {'candidates':>11} {'ms/candidate':>13}")
for n, t, c in rows:
    print(f"{n:>3} {t:>16.1f} {c:>11d} {t/c:>13.3f}")

xs = [float(r[0]) for r in rows]; ys = [r[1] for r in rows]
mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
b = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / sum((x-mx)**2 for x in xs)
a = my - b*mx
r2 = 1 - sum((y-(a+b*x))**2 for x,y in zip(xs,ys)) / sum((y-my)**2 for y in ys)
print(f"\nOLS  wall_ms(n) = {a:.1f} + {b:.3f} * n    R^2 = {r2:.4f}")

w20 = dict((r[0], r[1]) for r in rows)[20]
ncand = rows[0][2]
probe20 = b*20
print(f"\nAt the shipped n_samples=20:")
print(f"  checker wall (instrumented)          : {w20:.1f} ms  over {ncand} candidates")
print(f"  n-scaling (sensitivity-loop) portion : {probe20:.1f} ms = {probe20/w20*100:.1f}% of checker wall")
print(f"  n=0 intercept (everything else)      : {a:.1f} ms")
print(f"\n  => eliminating the sensitivity loop ENTIRELY (free replacement) is an")
print(f"     ABSOLUTE CEILING of -{probe20/w20*100:.1f}% checker wall.")

# corpus translation, using the same denominator the prior rounds used
CORPUS_S = 60.8
CHECKER_S = 9.89   # your_checker (full) 5.28 + (numeric only) 4.61, warm run
save_s = CHECKER_S * probe20/w20
print(f"\n  Corpus translation (warm run {CORPUS_S}s; perturbation-bearing checker")
print(f"  portion {CHECKER_S}s, same denominator as n_samples_curve/FINDINGS.md):")
print(f"     ceiling saving = {save_s:.2f} s = {save_s/CORPUS_S*100:.1f}% of corpus runtime")
