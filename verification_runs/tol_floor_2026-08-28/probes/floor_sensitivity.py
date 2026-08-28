"""
The 1e-6 tolerance floor: who it binds, what it protects, and what happens
to every banked verdict as it moves.

The floor is `adaptive_tol = max(3*P95(s), 1e-6)` (perturbation.py:210).
Every perturbation-family record in the current banked arms carries
`max_err` and `adaptive_tol` in its detail string (6-decimal fixed format),
so verdicts can be REPLAYED under any floor F:

    recorded tol > 1e-6  =>  unclamped u = recorded tol; new tol = max(u, F)
    recorded tol = 1e-6  =>  u <= 1e-6 unknown; bracket with u=0 and u=1e-6
                             (u=0 is exact for the J=0 ops where s == 0)

Both arms (A_lnfix, G_lnfix) are replayed; catch/FP are recomputed at the
check level (a mutant stays caught if ANY of its failing checks still
fails, gains a catch if a new check fails, etc. -- conservatively at the
perturbation-family level plus the unchanged non-family outcomes).

Also reports, per floor value, the size of exception category 1
(tol/F <= 1, the absolute arm of the resolvability criterion) and the
margin structure: distance from every record's max_err to the moving
boundary.

Run:  .venv/bin/python floor_sensitivity.py
"""
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(HERE, "..", "..", "layernorm_mask_fix_2026-08-28", "arms")

RE_ERR = re.compile(r"max_err=(\d+\.\d{6})")
RE_TOL = re.compile(r"adaptive_tol=(\d+\.\d{6})")
FLOOR0 = 1e-6
# print precision is 1e-6, so recorded-at-floor means tol prints as 0.000001
AT_FLOOR = 1.0000005e-6

FLOORS = [0.0, 1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def load(name):
    with gzip.open(os.path.join(ARMS, name), "rt") as f:
        return json.load(f)


def family_records(d):
    """(entry_idx, op, mutant, tag, check, max_err, tol, outcome) for every
    perturbation-family record; plus per-(entry, tag) the non-family fail
    count so overall verdicts can be recomposed."""
    fam, other_fail = [], {}
    for i, e in enumerate(d["entries"]):
        packs = [("mutant", e["mutant"]["records"])] + \
                [(f"ref{j}", r["records"]) for j, r in enumerate(e["refs"])]
        for tag, recs in packs:
            nf = 0
            for r in recs:
                det = r.get("detail") or ""
                me, mt = RE_ERR.search(det), RE_TOL.search(det)
                if me and mt and "adaptive_tol=" in det:
                    fam.append({"i": i, "op": e["op"],
                                "mutant": e["mutant"]["name"], "tag": tag,
                                "check": r["name"],
                                "err": float(me.group(1)),
                                "tol": float(mt.group(1)),
                                "outcome": r["outcome"]})
                elif r["outcome"] == "fail":
                    nf += 1
            other_fail[(i, tag)] = nf
    return fam, other_fail


def replay(fam, other_fail, F, u_at_floor):
    """Recompute per-(entry, tag) fail counts under floor F."""
    fails = dict.fromkeys(other_fail, 0)
    for r in fam:
        u = r["tol"] if r["tol"] > AT_FLOOR else u_at_floor
        newtol = max(u, F)
        if r["err"] > newtol:
            fails[(r["i"], r["tag"])] = fails[(r["i"], r["tag"])] + 1
    return {k: fails[k] + other_fail[k] for k in fails}


def main():
    for arm in ("A_lnfix.json.gz", "G_lnfix.json.gz"):
        d = load(arm)
        fam, other_fail = family_records(d)
        n_entries = len(d["entries"])
        at_floor = [r for r in fam if r["tol"] <= AT_FLOOR]
        print(f"\n=== {arm}: {len(fam)} perturbation-family records, "
              f"{len(at_floor)} floor-bound "
              f"({100*len(at_floor)/len(fam):.1f}%)")
        ops = {}
        for r in at_floor:
            ops[r["op"]] = ops.get(r["op"], 0) + 1
        print("    floor-bound by op:", dict(sorted(ops.items())))
        # margin structure on floor-bound records
        errs_ref = sorted(r["err"] for r in at_floor if r["tag"] != "mutant")
        errs_mut_fail = sorted(r["err"] for r in at_floor
                               if r["tag"] == "mutant" and r["outcome"] == "fail")
        errs_mut_pass = sorted(r["err"] for r in at_floor
                               if r["tag"] == "mutant" and r["outcome"] == "pass")
        nz = [e for e in errs_ref if e > 0]
        print(f"    floor-bound REF max_err: {len(errs_ref)} records, "
              f"nonzero {len(nz)}"
              + (f", largest {max(nz):.6f}" if nz else ""))
        print(f"    floor-bound MUTANT records: {len(errs_mut_fail)} failing "
              f"(smallest err {errs_mut_fail[0]:.6f})" if errs_mut_fail else
              "    floor-bound MUTANT records: none failing")
        if errs_mut_pass:
            pnz = [e for e in errs_mut_pass if e > 0]
            print(f"    floor-bound passing-mutant max_err: nonzero {len(pnz)}"
                  + (f", largest {max(pnz):.6f}" if pnz else ""))

        base = replay(fam, other_fail, FLOOR0, 0.0)

        print(f"    {'floor':>8} | {'catch':>6} {'FP':>4} | {'flips vs shipped':>17}"
              f" | {'cat-1 size':>10}")
        for F in FLOORS:
            res = {}
            for u in (0.0, FLOOR0):        # bracket the unknown unclamped tol
                fails = replay(fam, other_fail, F, u)
                caught = sum(1 for i in range(n_entries)
                             if fails.get((i, "mutant"), 0) > 0)
                fp = sum(1 for k, v in fails.items()
                         if k[1] != "mutant" and v > 0)
                flips = sum(1 for k in fails
                            if (fails[k] > 0) != (base[k] > 0))
                res[u] = (caught, fp, flips)
            cat1 = sum(1 for r in fam
                       if max((r["tol"] if r["tol"] > AT_FLOOR else 0.0), F) <= F)
            lo, hi = res[0.0], res[FLOOR0]
            same = lo == hi
            txt = (f"{lo[0]}/{n_entries} {lo[1]:>4} | {lo[2]:>17}" if same else
                   f"{lo[0]}-{hi[0]}/{n_entries} {lo[1]}-{hi[1]} | "
                   f"{lo[2]}-{hi[2]}")
            print(f"    {F:8.0e} | {txt} | {cat1:>10}")


if __name__ == "__main__":
    main()
