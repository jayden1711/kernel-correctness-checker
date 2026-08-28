"""
How much does the layer short-circuit ("early reject") in KernelChecker.run
actually save, and how often does it fire?

WHY THIS IS AN ANALYSIS AND NOT A BENCHMARK RUN
-----------------------------------------------
There is no "before" configuration to measure. The short-circuit is not a new
optimisation -- it is already present in `KernelChecker.run` (three gates at
verification/checker.py:114, 163, 171) and predates the 2026-08-20 layer
reorder. Measuring "with vs without" would mean deliberately breaking the
checker to time the broken version.

So this reconstructs the counterfactual from the warm corpus run already on
disk (`results_raw.json`, layer_convention `structural_algebraic_numeric_v2`).
That run records, per candidate, the `check_records` the pipeline actually
produced -- which tells us exactly which layer each candidate stopped at -- and
the same corpus was also run through three SINGLE-LAYER ablations, which give a
per-candidate cost for each layer in isolation.

Skipped cost per candidate is therefore read off the ablation runs at the same
index rather than modelled. Index alignment is sound: `harness.run()` iterates
`for name in systems: for entry in corpus:` and appends one mutant latency
followed by N_TRIALS_FPR reference latencies, so position i denotes the same
candidate in every system.

READ THE SAVINGS AS AN UPPER BOUND. Each ablation's per-candidate latency spans
input generation through verdict, so it carries fixed per-call overhead that a
skipped layer inside an already-running pipeline would not pay twice. The
overhead is not separable from the existing data; the true saving is somewhat
smaller than the figure printed here.

Reads only JSON -- no GPU, no torch, no triton. That matters: `triton` is a
linux-only dependency, so the corpus itself cannot even be imported on darwin,
which is why re-running the benchmark to answer this was not an option.

    python3 benchmarks/analyze_early_reject.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "benchmarks" / "autokernel" / "files" / "results_raw.json"

FULL = "your_checker (full)"
STRUCTURAL = "your_checker (structural only)"
ALGEBRAIC = "your_checker (algebraic only)"
NUMERIC = "your_checker (numeric only)"

# Which layer a candidate stopped at, keyed by the first failing layer.
STOPPED = {
    1: "early-reject @ L1 (structural)",
    2: "early-reject @ L2 (algebraic)",
    3: "ran all layers, failed in L3 (numeric)",
}
PASSED_ALL = "ran all layers, PASS"


def _first_failing_layer(record):
    """None if every check passed, else the lowest layer number that failed."""
    checks = record.get("check_records") or []
    if not checks:
        return "NO_RECORDS"
    failed = [c["layer"] for c in checks if c["outcome"] != "pass"]
    return min(failed) if failed else None


def _candidate_order(system):
    """
    Rebuild the (kind, record) sequence in the order harness.run() timed it:
    one mutant, then N_TRIALS_FPR references, per corpus entry.
    """
    mutants, refs = system["mutant_results"], system["ref_results"]
    n_trials = len(refs) // len(mutants)
    order = []
    for i, mutant in enumerate(mutants):
        order.append(("mutant", mutant))
        order.extend(("ref", refs[i * n_trials + j]) for j in range(n_trials))
    return order


def main():
    if not RAW.exists():
        sys.exit(f"missing {RAW} -- run the autokernel benchmark on a GPU first")

    raw = json.loads(RAW.read_text())
    missing = [s for s in (FULL, STRUCTURAL, ALGEBRAIC, NUMERIC) if s not in raw]
    if missing:
        sys.exit(f"results file lacks the ablation systems needed: {missing}")

    ms = {s: [x * 1000 for x in raw[s]["latencies"]] for s in
          (FULL, STRUCTURAL, ALGEBRAIC, NUMERIC)}
    order = _candidate_order(raw[FULL])
    if len(order) != len(ms[FULL]):
        sys.exit(f"index misalignment: {len(order)} candidates, {len(ms[FULL])} latencies")

    n = len(order)
    print(f"corpus: {n} candidates "
          f"({len(raw[FULL]['mutant_results'])} mutants + "
          f"{len(raw[FULL]['ref_results'])} reference trials)")
    print(f"layer_convention: {raw[FULL].get('layer_convention', 'ABSENT (pre-2026-08-20)')}\n")

    print("per-layer cost in isolation (single-layer ablation runs)")
    for label, system in (("structural", STRUCTURAL), ("algebraic", ALGEBRAIC),
                          ("numeric", NUMERIC), ("full pipeline", FULL)):
        vals = sorted(ms[system])
        print(f"  {label:14s} total {sum(vals) / 1000:6.3f}s   "
              f"mean {sum(vals) / len(vals):6.2f}ms   p50 {vals[len(vals) // 2]:6.2f}ms")

    buckets = {}
    saved_total = 0.0
    per_candidate = []
    for i, (kind, record) in enumerate(order):
        layer = _first_failing_layer(record)
        bucket = PASSED_ALL if layer is None else STOPPED.get(layer, str(layer))
        buckets.setdefault(bucket, []).append(i)

        # Cost of the layers this candidate never reached.
        if layer == 1:
            saved = ms[ALGEBRAIC][i] + ms[NUMERIC][i]
        elif layer == 2:
            saved = ms[NUMERIC][i]
        else:
            saved = 0.0
        saved_total += saved
        if saved:
            per_candidate.append((bucket, record.get("op"), record.get("mutant"),
                                  ms[FULL][i], saved))

    print("\nwhere each candidate stopped")
    for bucket in (STOPPED[1], STOPPED[2], STOPPED[3], PASSED_ALL):
        hits = len(buckets.get(bucket, []))
        print(f"  {hits:4d}  ({hits / n * 100:5.1f}%)  {bucket}")

    l1 = len(buckets.get(STOPPED[1], []))
    l2 = len(buckets.get(STOPPED[2], []))
    print(f"\n  early-reject fires on {l1 + l2}/{n} = {(l1 + l2) / n * 100:.1f}% of the corpus"
          f"  (structural {l1 / n * 100:.1f}%, algebraic {l2 / n * 100:.1f}%)")

    measured = sum(ms[FULL])
    counterfactual = measured + saved_total
    print("\nlatency impact (savings are an UPPER BOUND -- see module docstring)")
    print(f"  measured, with early-reject      {measured / 1000:6.3f}s   "
          f"mean {measured / n:6.2f}ms/candidate")
    print(f"  estimated, without early-reject  {counterfactual / 1000:6.3f}s   "
          f"mean {counterfactual / n:6.2f}ms/candidate")
    print(f"  delta                            {saved_total / 1000:6.3f}s   "
          f"{saved_total / counterfactual * 100:.1f}% of total")

    l1_saved = sum(s for b, _, _, _, s in per_candidate if b == STOPPED[1])
    print(f"\n  attributable to the STRUCTURAL gate alone: {l1_saved:.1f}ms "
          f"({l1_saved / counterfactual * 100:.2f}% of total)")
    print("  -- structural failures are rare, so the structural gate barely moves")
    print("     the aggregate; the algebraic gate does most of the skipping.")

    print(f"\nper-candidate detail ({len(per_candidate)} candidates that skipped work)")
    for bucket, op, mutant, full_ms, saved in sorted(per_candidate, key=lambda r: -r[4]):
        tag = "L1" if bucket == STOPPED[1] else "L2"
        print(f"  {tag}  {str(op):30s} {str(mutant):22s} "
              f"ran {full_ms:7.2f}ms   skipped ~{saved:7.2f}ms")


if __name__ == "__main__":
    main()
