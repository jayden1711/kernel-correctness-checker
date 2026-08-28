"""
Generic harness. Does NOT import your_checker or operators anymore --
it takes a `systems` dict and a `corpus` list as plain arguments, so it
works identically whether you're benchmarking my demo stand-in or your
real checker on your real 29-operator corpus.

CORPUS CONTRACT (see corpus_contract.py for a validator):
  Each entry is a dict with keys:
    op:          str, operator name
    mutant_name: str, identifies the specific bug
    ref_fn:      callable(*args) -> array-like, the correct reference
    mutant_fn:   callable(*args) -> array-like, the buggy candidate
    input_fn:    callable(rng) -> tuple of positional args for ref_fn/mutant_fn

SYSTEM CONTRACT:
  Each system is a callable(entry, is_mutant: bool, rng) -> (passed, dt, detail)
    passed: bool, True if the system judges the candidate CORRECT
    dt:     float, seconds elapsed (for latency reporting)
    detail: str | None, optional diagnostic (which stage/layer failed)
  See checker_adapter_template.py for how to wrap your real checker into
  this shape.
"""
import time
import numpy as np
from collections import defaultdict

from baselines import allclose_gate, autokernel_gate

N_TRIALS_FPR = 5  # re-check the reference this many times (some checks are stochastic)


def allclose_system(entry, is_mutant, rng):
    # TIMER SCOPE (see the convention note in run()): t0 goes here, before
    # input generation and both kernel invocations.
    #
    # This previously started the timer AFTER `ref_out`/`cand_out` were
    # computed, so it measured only the numpy comparison -- while every other
    # system measured a whole pipeline. That is what produced the "354x faster
    # than the full checker" figure in BENCHMARK_RESULTS.md §8.1: a
    # comparison-only measurement set against full-pipeline measurements.
    # allclose is genuinely the cheapest system here, but not by that factor.
    fn = entry["mutant_fn"] if is_mutant else entry["ref_fn"]
    t0 = time.perf_counter()
    args = entry["input_fn"](rng)
    ref_out = entry["ref_fn"](*args)
    cand_out = fn(*args)
    passed = allclose_gate(np.asarray(ref_out), np.asarray(cand_out))
    dt = time.perf_counter() - t0
    return passed, dt, None


def autokernel_system(entry, is_mutant, rng):
    fn = entry["mutant_fn"] if is_mutant else entry["ref_fn"]
    t0 = time.perf_counter()
    passed, failed_stage = autokernel_gate(
        entry["op"], entry["ref_fn"], fn, entry["input_fn"], rng)
    dt = time.perf_counter() - t0
    return passed, dt, failed_stage


# Always-available baselines. Add your real checker to this dict at the
# call site (see run_benchmark_template.py) -- don't hardcode it in here,
# so this file stays reusable across corpora / checker versions.
BASE_SYSTEMS = {
    "allclose": allclose_system,
    "autokernel_gate": autokernel_system,
}


def _call(system_fn, entry, is_mutant, rng):
    """
    Invoke a system and normalise its return to 4 elements.

    The system contract is (entry, is_mutant, rng) -> (passed, dt, detail),
    and every baseline still returns exactly that. Systems that also carry
    per-check attribution (see checker_adapter.py) return an optional 4th
    element: a list of {name, outcome, detail, subchecks} records. Unpacking
    by index rather than by tuple-destructuring is what lets both shapes
    coexist, so adding attribution to one system never touches the others.
    """
    out = system_fn(entry, is_mutant, rng)
    records = out[3] if len(out) > 3 else None
    return out[0], out[1], out[2], records


# Which layer-numbering convention the records in a results file use.
#
# On 2026-08-20 KernelChecker.run was reordered to structural -> algebraic ->
# numeric (numeric is ~13x the cost of algebraic and short-circuits are cheap),
# and the labels were swapped to match: algebraic became Layer 2, numeric
# Layer 3. A stored `layer: 2` therefore means NUMERIC in any artifact written
# before that date and ALGEBRAIC in anything after, with nothing else in the
# file to tell them apart.
#
# Scope note, so nobody over-reacts to this: NO current reader keys on the
# numeric layer value. analyze_check_ablation.py and layer_attribution.py both
# attribute by SYSTEM NAME ("your_checker (numeric only)"), which is unchanged.
# `layer` is used for display strings and carried into persisted records, so
# this marker exists for future analysis and for anyone reading old JSON or DB
# rows by hand -- not to repair a live breakage.
LAYER_CONVENTION = "structural_algebraic_numeric_v2"

def _warm(system_fn, entry, rng):
    """Run a system once per candidate, untimed, to warm the kernel cache.

    WHY: Triton JIT-compiles per (kernel function, tl.constexpr values). The
    checker's cross-shape sweep triggers combinations the cheaper baselines
    never touch, so whichever system reaches them FIRST paid every compile and
    every later system ran against a warm cache. Since `systems` is the outer
    loop below, that was decided by dict insertion order, not by cost.

    Measured on the 2026-08-20 run, before this existed: `your_checker (full)`
    (8th) totalled 43.4s against 18.3s for its three single-layer ablations
    (9th-11th) COMBINED -- a 25.1s excess, 58% of its measured time -- despite
    short-circuiting and therefore running a subset of their checks. It
    exceeded the ablation sum in 36 of 40 entries. `avg_pool1d/wrong_divisor`
    was 4218ms in `full` and 122ms across all three ablations: the same checks,
    35x cheaper purely for running second.

    Both `is_mutant` values are warmed because reference and mutant are
    distinct functions with independent compile caches; warming one would just
    relocate the bias to the other.

    THE RNG SNAPSHOT IS LOAD-BEARING, NOT HYGIENE. `input_fn(rng)` consumes
    draws (allclose_system above, checker_adapter._get_torch_inputs). Without
    restoring the bit generator's state, every subsequent input in the run
    would differ and CATCH RATES WOULD MOVE -- silently converting a latency
    fix into a correctness change. `.bit_generator.state` round-trips exactly;
    this is not a reseed. tests/instrumentation/check_item2_instrumentation.py
    carries a negative control proving removal is detectable.

    Exceptions are swallowed on purpose: a system that raises during warmup
    raises identically when timed, where the harness already models it as a
    four-valued `error` outcome. Aborting the run here would turn a handled
    condition into a crash.
    """
    state = _snapshot_rngs(rng)
    try:
        system_fn(entry, True, rng)
        system_fn(entry, False, rng)
    except Exception:
        pass
    finally:
        _restore_rngs(rng, state)


def _snapshot_rngs(rng):
    """Capture EVERY random source a warmup call can advance.

    Restoring numpy alone is not sufficient, and this was caught empirically
    rather than reasoned about: the first cold/warm comparison had identical
    verdicts but 4 of 319 check-level outcomes flipped in
    `your_checker (numeric only)` -- `cross_shape` and the
    `adversarial_*_rescaling` family. Those checks build their own inputs with
    `torch.randn` (verification/layer2_numeric_oracle/shape_generalization.py
    :73 and :125), which draws from torch's GLOBAL generator, not from the
    harness's numpy rng. Warmup advanced it and every later torch draw shifted.

    Structural and algebraic ablations were unaffected, which is why the gap
    was easy to miss: their checks test gross properties that do not care about
    the exact random values, while the numeric layer's adaptive tolerance
    flips at the margin.

    torch is imported lazily and failures are ignored so harness.py stays
    importable under tests/instrumentation's stubbed torch.
    """
    state = {"np": rng.bit_generator.state}
    try:
        import torch
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def _restore_rngs(rng, state):
    rng.bit_generator.state = state["np"]
    try:
        import torch
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        pass


def run(systems: dict, corpus: list, seed: int = 0, warmup: bool = True):
    """Run every system over every corpus entry.

    TIMER CONVENTION: each system starts its timer immediately after selecting
    the candidate function, so input generation, kernel invocation and the
    verdict are all inside the measured region. Three different conventions
    were in use before 2026-08-20 -- `allclose` timed only its numpy
    comparison, which is what produced the "354x faster" figure by comparing a
    comparison-only measurement against full-pipeline ones.

    `warmup=False` reproduces the old cold-cache measurements for comparison;
    it does not change any verdict either way.
    """
    rng = np.random.default_rng(seed)
    results = {name: {"mutant_results": [], "ref_results": [], "latencies": [],
                      "layer_convention": LAYER_CONVENTION}
               for name in systems}

    for name, system_fn in systems.items():
        for entry in corpus:
            if warmup:
                _warm(system_fn, entry, rng)
            passed, dt, detail, records = _call(system_fn, entry, True, rng)
            results[name]["mutant_results"].append({
                "op": entry["op"], "mutant": entry["mutant_name"],
                "caught": (not passed), "detail": detail,
                "check_records": records,
            })
            results[name]["latencies"].append(dt)

            for _ in range(N_TRIALS_FPR):
                ref_passed, ref_dt, ref_detail, ref_records = _call(system_fn, entry, False, rng)
                results[name]["ref_results"].append({
                    "op": entry["op"], "false_positive": (not ref_passed),
                    "detail": ref_detail,
                    "check_records": ref_records,
                })
                results[name]["latencies"].append(ref_dt)

    return results


def _percentile(sorted_values, p):
    """Linear-interpolated percentile of an ALREADY-SORTED list.

    Matches `numpy.percentile`'s default (`method="linear"`), so p50 equals the
    median. Implemented in pure Python on purpose, for two reasons:

      * `summarize()` is exercised by tests/instrumentation/ under a stubbed
        numpy. Calling `np.percentile` there would mean either teaching the stub
        to reimplement numpy -- making the test validate the stub rather than
        this code -- or losing that coverage. See SESSION_HANDOFF.md §5
        instance 7 for why that shape of "test" is worth avoiding.
      * numpy's percentile keyword has changed across versions
        (`interpolation=` -> `method=`). Spelling the interpolation out here
        pins the semantics rather than inheriting a shifting default.

    Verified against numpy's linear method on the 2026-08-20 corpus run:
    your_checker (full) -> p50 32.34, p90 472.57, p99 2248.94 ms.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    pos = (p / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def summarize(results):
    summary = {}
    for name, r in results.items():
        n_mutants = len(r["mutant_results"])
        n_caught = sum(x["caught"] for x in r["mutant_results"])
        n_refs = len(r["ref_results"])
        n_fp = sum(x["false_positive"] for x in r["ref_results"])
        mean_latency_ms = 1000 * np.mean(r["latencies"]) if r["latencies"] else 0.0

        # Percentiles alongside the mean, because the mean alone is misleading
        # here and was being cited as if it were typical. Measured on the
        # 2026-08-20 corpus run, mean/p50 ratios: your_checker (full) 5.6x,
        # autokernel_gate (faithful) 33.5x, gpuemu (boundary_shape) 21.9x. The
        # distribution is heavy-tailed, not warmup noise -- for your_checker
        # (full), the top 10 of 240 trials are 49% of total time and the top 24
        # are 73%. A mean over that says little about a typical check, and
        # optimising against it would target the wrong thing.
        #
        # ADDITIVE ONLY: `mean_latency_ms` above is deliberately left exactly as
        # it was so this run stays numerically comparable to every earlier one.
        # These are new keys, not a replacement.
        if r["latencies"]:
            _ms = sorted(1000 * x for x in r["latencies"])
            p50_latency_ms = _percentile(_ms, 50)
            p90_latency_ms = _percentile(_ms, 90)
            p99_latency_ms = _percentile(_ms, 99)
            max_latency_ms = _ms[-1]
        else:
            p50_latency_ms = p90_latency_ms = p99_latency_ms = max_latency_ms = 0.0

        per_op = defaultdict(lambda: {"caught": 0, "total": 0})
        for x in r["mutant_results"]:
            per_op[x["op"]]["total"] += 1
            per_op[x["op"]]["caught"] += int(x["caught"])

        # Which specific mutants each system MISSED -- the most actionable
        # output for debugging your checker, not just the aggregate score.
        missed = [f"{x['op']}/{x['mutant']}" for x in r["mutant_results"] if not x["caught"]]

        # Same idea for false positives: aggregate FP rate alone doesn't say
        # WHICH operator/check is rejecting the correct reference, so break
        # it down per-op too, plus a few example failure details to root-cause
        # a nonzero rate instead of just knowing it's nonzero.
        per_op_fp = defaultdict(lambda: {"fp": 0, "total": 0})
        fp_detail_samples = defaultdict(set)
        for x in r["ref_results"]:
            per_op_fp[x["op"]]["total"] += 1
            per_op_fp[x["op"]]["fp"] += int(x["false_positive"])
            if x["false_positive"] and x["detail"]:
                fp_detail_samples[x["op"]].add(str(x["detail"])[:200])

        summary[name] = {
            "catch_rate": n_caught / n_mutants if n_mutants else float("nan"),
            "false_positive_rate": n_fp / n_refs if n_refs else float("nan"),
            "mean_latency_ms": mean_latency_ms,
            "p50_latency_ms": p50_latency_ms,
            "p90_latency_ms": p90_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "max_latency_ms": max_latency_ms,
            "per_op_catch_rate": {op: v["caught"] / v["total"] for op, v in per_op.items()},
            "per_op_false_positive_rate": {
                op: v["fp"] / v["total"] for op, v in per_op_fp.items() if v["fp"] > 0
            },
            "false_positive_detail_samples": {
                op: sorted(details)[:3] for op, details in fp_detail_samples.items()
            },
            "missed_mutants": missed,
            "n_mutants": n_mutants,
            "n_fp_checks": n_refs,
        }
    return summary
