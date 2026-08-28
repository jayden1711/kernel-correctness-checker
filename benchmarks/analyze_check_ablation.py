"""
Per-check ablation tables for Layer 2 and Layer 3, built from the raw
benchmark records that run_benchmark.py now persists to results_raw.json.

WHY THIS EXISTS
---------------
BENCHMARK_RESULTS.md reports Layer 2 as one aggregate 100%/0% and Layer 3 as
one aggregate 45%. Neither says which individual checks are doing the work,
which are redundant with each other, or which have never caught anything --
the advisor's "tests scattered, analyze ablation for each check" note.

The attribution was already being computed on every run and thrown away:
harness.summarize() keeps rates / missed_mutants / FP samples and drops the
per-mutant detail entirely. checker_adapter.py now emits structured per-check
records, harness.py carries them, run_benchmark.py writes them out, and this
script reads them.

WHY THE ABLATIONS AND NOT THE FULL CHECKER
------------------------------------------
Attribution comes from "your_checker (numeric only)" and "your_checker
(algebraic only)", never from "your_checker (full)". KernelChecker.run
short-circuits between layers (checker.py:114/155/213), so in a full run a
Layer-2 check only executes when every Layer-1 check passed -- its catch
counts would be conditioned on Layer 1 missing. The single-layer ablations
run every check in their layer unconditionally, which is exactly why
checker_adapter.py builds them that way. Full-checker records are tagged
short_circuited=True and are ignored here.

WHAT COUNTS AS A CATCH
----------------------
Only outcome == "fail". An "error" (the check raised) is NOT a catch, and is
reported in its own column. This distinction is the whole reason the records
are four-valued: in the AutoKernel baseline audit an identical bare-except
pattern scored a ValueError as a legitimate gate failure and produced that
baseline's entire 18% false-positive rate. A check that crashes on a mutant
must not look like a check that detected it.

Usage:
    python3 benchmarks/analyze_check_ablation.py [path/to/results_raw.json]
    # writes benchmarks/CHECK_ABLATION.md
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW = os.path.join(HERE, "autokernel", "files", "results_raw.json")
OUT_PATH = os.path.join(HERE, "CHECK_ABLATION.md")

NUMERIC_SYSTEM = "your_checker (numeric only)"
ALGEBRAIC_SYSTEM = "your_checker (algebraic only)"
STRUCTURAL_SYSTEM = "your_checker (structural only)"

# Fixed Layer-2 checks the code can emit. Listed explicitly so a check that
# NEVER APPEARS in the data is distinguishable from one that appears and
# never catches -- "absent" and "present but useless" are different findings
# and only the roster makes the first one visible. backward_pass is the live
# example: all 29 specs set requires_backward=False (base_spec.py defaults it
# to True), so check_backward_pass never executes anywhere in this corpus.
LAYER2_FIXED_CHECKS = [
    "output_shape",
    "perturbation_tolerance",
    "cross_shape",
    "weight_magnitude",
    "backward_pass",
]


def _key(rec):
    return f"{rec['op']}/{rec['mutant']}"


def _expand(records):
    """Yield (check_name, outcome) per record, expanding compound checks.

    cross_shape and weight_magnitude return per-sub-check outcomes as a third
    element; those surface as "parent[sub]" so a 5-shape sweep and a 4-variant
    magnitude probe are individually attributable. The parent is still yielded
    so the aggregate stays visible alongside its parts.

    `subchecks` is validated here rather than trusted. checker_adapter._try
    now guarantees a list-or-None, but this script also reads results_raw.json
    files produced by EARLIER runs, which can still contain a non-list in that
    slot (tile_coverage's int column count -- see that function's docstring).
    Without this guard those files crash the whole report with
    `TypeError: 'int' object is not iterable`, losing attribution for all 94
    checks because of 2 malformed records. A malformed slot means "not a
    compound check", so the parent is still yielded and only the (absent)
    sub-records are skipped.
    """
    for r in records or []:
        if r.get("short_circuited"):
            continue
        yield r["name"], r["outcome"]
        subs = r.get("subchecks")
        if not isinstance(subs, list):
            continue
        for sub in subs:
            yield f"{r['name']}[{sub['name']}]", sub["outcome"]


def collect(system_results):
    """-> (stats, catch_sets, mutants_seen)

    stats[check] = {pass, fail, error, skip, fp}
    catch_sets[check] = set of "op/mutant" the check caught
    """
    stats = defaultdict(lambda: dict(**{"pass": 0, "fail": 0, "error": 0, "skip": 0, "fp": 0}))
    catch_sets = defaultdict(set)
    mutants_seen = set()

    for rec in system_results.get("mutant_results", []):
        mutants_seen.add(_key(rec))
        for name, outcome in _expand(rec.get("check_records")):
            stats[name][outcome] = stats[name].get(outcome, 0) + 1
            if outcome == "fail":
                catch_sets[name].add(_key(rec))

    # A check that fails on the CORRECT reference is a false-positive source.
    # Aggregate FP rate alone never says which check is responsible; this does.
    for rec in system_results.get("ref_results", []):
        for name, outcome in _expand(rec.get("check_records")):
            if outcome == "fail":
                stats[name]["fp"] += 1

    return stats, catch_sets, mutants_seen


def table_per_check(stats, catch_sets, title, roster=None):
    lines = [f"### {title}", "",
             "| Check | ran | caught | catch rate | errors | skips | FPs on reference |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(stats, key=lambda n: (-stats[n]["fail"], n)):
        st = stats[name]
        ran = st["pass"] + st["fail"]
        rate = f"{100.0 * st['fail'] / ran:.0f}%" if ran else "--"
        lines.append(f"| `{name}` | {ran} | {st['fail']} | {rate} | "
                     f"{st['error']} | {st['skip']} | {st['fp']} |")
    if roster:
        absent = [c for c in roster if c not in stats]
        for name in absent:
            lines.append(f"| `{name}` | 0 | 0 | -- | 0 | 0 | 0 |  <!-- never ran -->")
    lines.append("")
    return lines


def zero_catch_report(stats, roster=None):
    never_ran, ran_never_caught = [], []
    for name, st in stats.items():
        if st["fail"]:
            continue
        (never_ran if (st["pass"] + st["fail"]) == 0 else ran_never_caught).append(name)
    if roster:
        never_ran += [c for c in roster if c not in stats]

    lines = ["### Zero-catch checks", ""]
    lines.append("**Never ran** (no pass/fail outcome recorded -- dead code on this "
                 "corpus, or skipped every time). Removing these changes nothing; "
                 "keeping them requires a mutant or a spec that actually exercises them.")
    lines.append("")
    lines += [f"- `{n}`" for n in sorted(set(never_ran))] or ["- _(none)_"]
    lines.append("")
    lines.append("**Ran but never caught anything.** These execute on every relevant "
                 "mutant and have never once been the check that found a bug. Each is "
                 "a candidate for removal, or evidence that the corpus lacks a mutant "
                 "targeting it -- the two are indistinguishable from this table alone "
                 "and need a per-check judgement.")
    lines.append("")
    lines += [f"- `{n}`" for n in sorted(ran_never_caught)] or ["- _(none)_"]
    lines.append("")
    return lines


def cocatch_report(catch_sets, min_catches=1):
    """Pairwise overlap between checks that catch anything.

    Two checks that catch identical mutant sets are redundant ON THIS CORPUS
    -- which is a claim about the corpus as much as about the checks, and is
    stated that way rather than as "one of these is useless".
    """
    active = {k: v for k, v in catch_sets.items() if len(v) >= min_catches}
    names = sorted(active, key=lambda n: (-len(active[n]), n))
    lines = ["### Redundancy: pairwise catch overlap", ""]
    if len(names) < 2:
        lines += ["_Fewer than two checks caught anything; no overlap to report._", ""]
        return lines

    lines.append("| Check A | Check B | A only | both | B only | relationship |")
    lines.append("|---|---|---:|---:|---:|---|")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = active[a], active[b]
            both = len(A & B)
            if not both:
                continue
            rel = ("identical" if A == B else
                   f"`{a}` subsumes `{b}`" if B < A else
                   f"`{b}` subsumes `{a}`" if A < B else "partial overlap")
            lines.append(f"| `{a}` | `{b}` | {len(A - B)} | {both} | {len(B - A)} | {rel} |")
    lines.append("")
    lines.append("`identical` pairs are the actionable redundancy: on this corpus one "
                 "of the two never contributes a catch the other misses.")
    lines.append("")
    return lines


def layer3_per_spec(system_results):
    """(op, property) -> outcome counts. Layer 3 is per-operator by
    construction, so the aggregate 45% hides both which properties work and
    which operators have no property coverage at all."""
    per = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0, "skip": 0})
    ops_seen = defaultdict(set)
    for rec in system_results.get("mutant_results", []):
        op = rec["op"]
        ops_seen[op].add(rec["mutant"])
        for name, outcome in _expand(rec.get("check_records")):
            per[(op, name)][outcome] += 1

    lines = ["### Layer 3 -- per operator spec x property", "",
             "| Operator | Property | mutants | caught | errors |",
             "|---|---|---:|---:|---:|"]
    for (op, name) in sorted(per):
        st = per[(op, name)]
        lines.append(f"| {op} | `{name}` | {len(ops_seen[op])} | {st['fail']} | {st['error']} |")
    lines.append("")

    no_props = sorted(op for op in ops_seen if not any(k[0] == op for k in per))
    if no_props:
        lines.append("**Operators with no algebraic properties at all** -- Layer 3 "
                     "cannot catch their mutants under any circumstances:")
        lines.append("")
        lines += [f"- `{op}`" for op in no_props]
        lines.append("")

    rolled = defaultdict(lambda: {"specs": 0, "fail": 0, "error": 0})
    for (op, name), st in per.items():
        rolled[name]["specs"] += 1
        rolled[name]["fail"] += st["fail"]
        rolled[name]["error"] += st["error"]
    lines += ["### Layer 3 -- rolled up by property name", "",
              "| Property | specs using it | total catches | errors |",
              "|---|---:|---:|---:|"]
    for name in sorted(rolled, key=lambda n: (-rolled[n]["fail"], n)):
        r = rolled[name]
        lines.append(f"| `{name}` | {r['specs']} | {r['fail']} | {r['error']} |")
    lines.append("")
    lines.append("A property used by many specs with zero total catches is doing no "
                 "work anywhere, not just on one operator.")
    lines.append("")
    return lines


def consistency_check(system_results, catch_sets, label):
    """The set of mutants with >=1 'fail' record must equal the set the harness
    scored as caught. A mismatch means the instrumentation is miscounting --
    most likely by folding an 'error' into a 'fail', which is precisely the
    bug class this whole design exists to avoid."""
    from_records = set()
    for s in catch_sets.values():
        from_records |= s
    from_verdict = {_key(r) for r in system_results.get("mutant_results", []) if r["caught"]}
    lines = [f"### Consistency check -- {label}", ""]
    if from_records == from_verdict:
        lines.append(f"OK: {len(from_verdict)} mutants caught, and the per-check "
                     "records agree exactly with the harness verdict.")
    else:
        lines.append("**MISMATCH -- do not trust the tables above.**")
        lines.append("")
        lines.append(f"- caught per harness verdict but no failing record: "
                     f"`{sorted(from_verdict - from_records)}`")
        lines.append(f"- failing record but not scored as caught: "
                     f"`{sorted(from_records - from_verdict)}`")
        lines.append("")
        lines.append("The usual cause is a check whose only non-pass outcome was "
                     "`error`: the harness verdict counts it as a failure (legacy "
                     "bool coercion) while the records correctly do not count it as "
                     "a catch. That gap is a real finding, not a script bug -- it "
                     "means a mutant's 'catch' was a crash.")
    lines.append("")
    return lines


def main():
    raw_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW
    if not os.path.exists(raw_path):
        sys.exit(f"No results_raw.json at {raw_path}\n"
                 "Run benchmarks/autokernel/files/run_benchmark.py on a CUDA "
                 "runtime first -- this corpus is real Triton kernels with no CPU path.")

    with open(raw_path) as f:
        raw = json.load(f)

    missing = [s for s in (NUMERIC_SYSTEM, ALGEBRAIC_SYSTEM) if s not in raw]
    if missing:
        sys.exit(f"results_raw.json is missing required systems: {missing}\n"
                 f"present: {sorted(raw)}")

    out = ["# Per-check ablation: Layer 2 and Layer 3", "",
           "Generated by `benchmarks/analyze_check_ablation.py` from "
           "`results_raw.json`. Catches come from the single-layer ablations, "
           "which run every check in their layer unconditionally; the "
           "short-circuiting full checker is deliberately not used for "
           "attribution. Only `fail` counts as a catch -- `error` means the "
           "check raised and is reported separately.", ""]

    n_stats, n_catch, n_mut = collect(raw[NUMERIC_SYSTEM])
    out += ["---", "", "## Layer 2 (numeric)", ""]
    out += [f"{len(n_mut)} mutants, {len(n_stats)} distinct checks observed.", ""]
    out += table_per_check(n_stats, n_catch, "Per-check catch rate", roster=LAYER2_FIXED_CHECKS)
    out += zero_catch_report(n_stats, roster=LAYER2_FIXED_CHECKS)
    out += cocatch_report(n_catch)
    out += consistency_check(raw[NUMERIC_SYSTEM], n_catch, "Layer 2")

    a_stats, a_catch, a_mut = collect(raw[ALGEBRAIC_SYSTEM])
    out += ["---", "", "## Layer 3 (algebraic)", ""]
    out += layer3_per_spec(raw[ALGEBRAIC_SYSTEM])
    out += cocatch_report(a_catch)
    out += consistency_check(raw[ALGEBRAIC_SYSTEM], a_catch, "Layer 3")

    if STRUCTURAL_SYSTEM in raw:
        s_stats, s_catch, _ = collect(raw[STRUCTURAL_SYSTEM])
        out += ["---", "", "## Layer 1 (structural) -- for completeness", ""]
        out += table_per_check(s_stats, s_catch, "Per-check catch rate")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(out))
    print(f"Wrote {OUT_PATH}")
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
