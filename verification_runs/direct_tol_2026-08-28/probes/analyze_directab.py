"""
Offline analysis of the DIRECT-tolerance A/B campaign.

Inputs (downloaded from the T4's /content/directab):
  data/gpu/{A_ver,D_ver}.json           verdict/attribution passes (timing on)
  data/gpu/{A,D}_w{1..5}.json           wall-clock reps (timing off)
  data/gpu/nm_{mA,mD}.json              m-series pert response per arm
  data/gpu/nm_{vA,vD}.json              v-series verdict response per arm
  data/preregistered_direct_predictions.json   banked BEFORE the run

Outputs: printed report (bank to data/analysis.log).

Run:  .venv/bin/python analyze_directab.py
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
GPU = os.path.join(DATA, "gpu")


def load(name):
    return json.load(open(os.path.join(GPU, name)))


def summary(d):
    s = d["summary"]
    return f"catch {s['n_caught']}/{s['n_mutants']} fp {s['n_fp']}/{s['n_ref']}"


def wall(d):
    tot = 0.0
    for e in d["entries"]:
        tot += e["mutant"]["dt_ms"]
        for r in e["refs"]:
            tot += r["dt_ms"]
    return tot / 1000.0


def outcomes(d):
    out = {}
    for e in d["entries"]:
        key = (e["op"], e["mutant"]["name"])
        out[key + ("mutant",)] = e["mutant"]["caught"]
        for i, r in enumerate(e["refs"]):
            out[key + (f"ref{i}",)] = r["false_positive"]
    return out


def failing_sets(d):
    out = {}
    for e in d["entries"]:
        key = (e["op"], e["mutant"]["name"])
        for tag, rec in [("mutant", e["mutant"])] + [
                (f"ref{i}", r) for i, r in enumerate(e["refs"])]:
            fails = tuple(sorted(x["name"] for x in rec["records"]
                                 if x["outcome"] == "fail"))
            out[key + (tag,)] = fails
    return out


def pert_records(d):
    """(op, mutant, trial, check) -> duration_ms + estimator marker."""
    rows = []
    for e in d["entries"]:
        for tag, rec in [("mutant", e["mutant"])] + [
                (f"ref{i}", r) for i, r in enumerate(e["refs"])]:
            for x in rec["records"]:
                nm = x["name"]
                if nm == "perturbation_tolerance" or nm.startswith(
                        "adversarial"):
                    det = str(x.get("detail", ""))
                    est = ("direct" if "closed-form" in det else "probe")
                    rows.append({"op": e["op"], "trial": tag, "check": nm,
                                 "ms": x.get("duration_ms"), "est": est,
                                 "outcome": x["outcome"]})
    return rows


def main():
    print("==== 1. catch / FP per arm ====")
    A = load("A_ver.json")
    D = load("D_ver.json")
    print("A_ver:", summary(A))
    print("D_ver:", summary(D))

    print("\n==== 2. verdict + attribution identity ====")
    oa, od = outcomes(A), outcomes(D)
    diff = [k for k in oa if oa[k] != od.get(k)]
    print(f"verdict diffs: {len(diff)}")
    for k in diff[:20]:
        print("  ", k, "A:", oa[k], "D:", od.get(k))
    fa, fd = failing_sets(A), failing_sets(D)
    fdiff = [k for k in fa if fa[k] != fd.get(k)]
    print(f"failing-set diffs: {len(fdiff)}")
    for k in fdiff[:20]:
        print("  ", k, "\n     A:", fa[k], "\n     D:", fd.get(k))

    print("\n==== 3. estimator coverage in D ====")
    rows = pert_records(D)
    n_direct = sum(r["est"] == "direct" for r in rows)
    print(f"perturbation-routed records: {len(rows)}; "
          f"direct: {n_direct} ({n_direct/len(rows):.1%}); "
          f"probe fallback: {len(rows)-n_direct}")
    bycheck = defaultdict(lambda: [0, 0])
    for r in rows:
        bycheck[r["op"]][0] += r["est"] == "direct"
        bycheck[r["op"]][1] += 1
    fallback_ops = {op: c for op, c in bycheck.items() if c[0] < c[1]}
    print("ops with any probe fallback:",
          {op: f"{c[0]}/{c[1]}" for op, c in fallback_ops.items()})

    print("\n==== 4. per-check serialised timing (shares only) ====")
    for arm, d in (("A", A), ("D", D)):
        rows = pert_records(d)
        ms = [r["ms"] for r in rows if r["ms"] is not None]
        print(f"  {arm}: perturbation-routed median "
              f"{np.median(ms):.2f} ms  p90 {np.percentile(ms,90):.2f} ms  "
              f"total {sum(ms)/1000:.2f} s")

    print("\n==== 5. wall clock (timing flag OFF, interleaved reps) ====")
    walls = {"A": [], "D": []}
    for arm in ("A", "D"):
        for i in range(1, 6):
            try:
                walls[arm].append(wall(load(f"{arm}_w{i}.json")))
            except FileNotFoundError:
                pass
    for arm in ("A", "D"):
        w = walls[arm]
        print(f"  {arm}: reps {[f'{x:.2f}' for x in w]}  "
              f"median {np.median(w):.2f} s")
    if walls["A"] and walls["D"]:
        med_a, med_d = np.median(walls["A"]), np.median(walls["D"])
        print(f"  DELTA (checker wall): {med_a - med_d:+.2f} s = "
              f"{(med_a - med_d)/med_a:+.2%} of A")
        # spread-based noise floor
        span_a = (max(walls['A']) - min(walls['A'])) / med_a
        span_d = (max(walls['D']) - min(walls['D'])) / med_d
        print(f"  rep spread: A {span_a:.1%}  D {span_d:.1%}")

    print("\n==== 6. near-miss m-series response per arm ====")
    prereg = json.load(open(os.path.join(
        DATA, "preregistered_direct_predictions.json")))
    for tag in ("mA", "mD"):
        try:
            nm = json.load(open(os.path.join(GPU, f"nm_{tag}.json")))
        except FileNotFoundError:
            print(f"  {tag}: MISSING")
            continue
        recs = [r for r in nm["records"] if "margin" in r]
        print(f"  {tag}:")
        for op in sorted({r["op"] for r in recs}):
            row = []
            for mk in ("m050", "m080", "m100", "m125", "m200"):
                sel = [not r["pert_passed"] for r in recs
                       if r["op"] == op and r["mutant"] == mk]
                row.append(f"{sum(sel)}/{len(sel)}")
            extra = ""
            if tag == "mD":
                pred = [f"{prereg[op][mk]:.0%}" for mk in
                        ("m050", "m080", "m100", "m125", "m200")]
                extra = f"   prereg pred: {pred}"
            print(f"    {op:14s} {row}{extra}")

    print("\n==== 7. near-miss v-series response per arm ====")
    for tag in ("vA", "vD"):
        try:
            nm = json.load(open(os.path.join(GPU, f"nm_{tag}.json")))
        except FileNotFoundError:
            print(f"  {tag}: MISSING")
            continue
        recs = nm["records"]
        print(f"  {tag}:")
        for op in sorted({r["op"] for r in recs}):
            row = []
            for mk in ("v050", "v080", "v100", "v125", "v200"):
                sel = [r["caught"] for r in recs
                       if r["op"] == op and r["mutant"] == mk]
                row.append(f"{sum(sel)}/{len(sel)}")
            print(f"    {op:14s} {row}")


if __name__ == "__main__":
    main()
