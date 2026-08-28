"""
The BEST case the structural path could possibly have, from banked GPU data.

The cost probe shows M3 -- the simulation that actually produces `tol` -- is
1128x the Monte-Carlo path it replaces. But M3's input is `w = profile/L`, and
for the 9 shape-only operators that vector is a CONSTANT given the output
shape. So `y` is cacheable per (op, shape) for those 9, and their structural
cost amortises to ~0 across a 240-candidate run.

That is the most favourable framing available -- structural L + cached M3 on
the 9 shape-only operators, Monte-Carlo everywhere else -- and it is the one
worth costing, because if THAT is small the general case cannot be large.

Denominator discipline is the same as n_samples_curve/FINDINGS.md: per-check
shares come from the KCC_CHECK_TIMING arms (CUDA serialised, so shares are
meaningful and absolutes are upper bounds), and are translated against the
9.89 s perturbation-bearing portion of the 60.8 s warm corpus run.
"""
import gzip, json

ARMS = "verification_runs/n_samples_curve_2026-08-25/arms"
NS = [3, 5, 10, 15, 20, 40]
EXACT = {"argmax", "argmin"}
STATIC = {"sum_reduction", "mean_reduction", "max_reduction", "min_reduction",
          "max_pool1d", "max_pool2d", "max_pool3d",
          "avg_pool1d", "avg_pool2d", "avg_pool3d"}
CORPUS_S, CHECKER_S = 60.8, 9.89


def buckets(n):
    d = json.load(gzip.open(f"{ARMS}/VALID_n{n}.json.gz"))
    out = {"static": 0.0, "dynamic": 0.0, "ordinary_static": 0.0,
           "ordinary_dynamic": 0.0, "adv_static": 0.0, "adv_dynamic": 0.0,
           "all": 0.0}
    for e in d["entries"]:
        op = e["op"]
        recs = [r for r in e["mutant"]["records"]] + \
               [r for rf in e["refs"] for r in rf["records"]]
        for r in recs:
            ms = r.get("duration_ms") or 0.0
            out["all"] += ms
            if op in EXACT:
                continue
            nm = r["name"]
            if nm == "perturbation_tolerance":
                kind = "ordinary"
            elif nm.startswith("adversarial_") and nm != "adversarial_setup":
                kind = "adv"
            else:
                continue
            grp = "static" if op in STATIC else "dynamic"
            out[grp] += ms
            out[f"{kind}_{grp}"] += ms
    return out


# slope of each bucket in n == the probing (sensitivity-loop) cost in it
def slope(key):
    xs = [float(n) for n in NS]
    ys = [B[n][key] for n in NS]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


B = {n: buckets(n) for n in NS}
all20 = B[20]["all"]

print("Probing (n-scaling) cost at the shipped n_samples=20, by bucket:\n")
print(f"{'bucket':<40}{'ms':>10}{'% of check time':>18}")
tot = 0.0
res = {}
for key in ("ordinary_static", "adv_static", "ordinary_dynamic", "adv_dynamic"):
    v = slope(key) * 20
    res[key] = v
    tot += v
    print(f"{key:<40}{v:>10.1f}{v/all20*100:>17.1f}%")
print(f"{'TOTAL probing':<40}{tot:>10.1f}{tot/all20*100:>17.1f}%")
print()

WALL20 = 8474.3   # instrumented checker wall at n=20, banked


def corpus(ms):
    frac = ms / WALL20
    return frac, CHECKER_S * frac, CHECKER_S * frac / CORPUS_S * 100


print("Ceiling if the replacement were FREE (it is not -- see cost_probe.py):\n")
print(f"{'scenario':<52}{'checker wall':>14}{'corpus':>10}")
scen = [
    ("A  all 27 ops, ordinary + adversarial inputs", tot),
    ("B  all 27 ops, ORDINARY inputs only (validated)",
     res["ordinary_static"] + res["ordinary_dynamic"]),
    ("C  9 shape-only ops, cached M3, ordinary+adv",
     res["ordinary_static"] + res["adv_static"]),
    ("D  9 shape-only ops, cached M3, ORDINARY only",
     res["ordinary_static"]),
]
for nm, ms in scen:
    f, s, pc = corpus(ms)
    print(f"{nm:<52}{-f*100:>13.1f}%{-pc:>9.1f}%")
print()
print("A is the arithmetic ceiling and assumes a zero-cost estimator.")
print("D is the only scenario that is BOTH cheaper than probing AND inside")
print("the regime the closed forms were validated on.")
