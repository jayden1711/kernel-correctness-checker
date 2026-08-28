"""
Wires the real three-layer KernelChecker (verification/) into the harness
contract: (entry, is_mutant, rng) -> (passed, dt, detail).

`my_checker_system` (the full checker) just calls the real
verification.checker.KernelChecker exactly as run_checker.py does.

The three ablation variants (numeric_only / algebraic_only /
structural_only) can't just call KernelChecker.run() with early-return
disabled, because that method short-circuits between layers (layer 2 never
runs if layer 1 failed) -- which is correct for the real checker but wrong
for an ablation, whose whole point is "how good is this layer ALONE,
independent of whether the others would have caught it first". So each
ablation replicates exactly the corresponding block of
verification/checker.py's KernelChecker.run(), using the same underlying
check functions, but run unconditionally.

numeric_only explicitly includes spec.get_adversarial_inputs(inputs) --
softmax's adversarial generators (max_in_last_tile, equal_logits,
extreme_range, non_power_of_two, near_zero_variance) and every other
operator's spec-declared adversarial variants -- not just a single
random-input perturbation check. That adversarial loop is genuinely part
of Layer 2 in checker.py (see its "Adversarial inputs" section), so
leaving it out here would understate what the numeric layer alone catches.
"""

import time

import torch

from verification.checker import KernelChecker, _check_cross_shape, _check_exact_match
from verification.layer1_structural.ast_analysis import (
    check_ghost_optimization,
    check_partial_computation,
    check_timing_manipulation,
)
from verification.layer1_structural.runtime_guards import (
    check_determinism,
    check_dtype_preserved,
    check_kernel_executed,
    check_nan_inf,
)
from verification.layer1_structural.tile_coverage import (
    check_all_tiles_visited,
    check_all_tiles_visited_generic,
)
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance
from verification.layer2_numeric_oracle.shape_generalization import (
    check_backward_pass,
    check_output_shape,
    check_weight_magnitude,
)


def _try(name, fn):
    """
    Run one check and return (name, passed, detail, record).

    CRITICAL -- `passed` and `detail` keep the ORIGINAL bool-coercion
    semantics, deliberately unchanged by this instrumentation:

      * an exception still yields passed=False
      * a check returning None (a SKIP) still yields passed=False via
        bool(None)

    Both are wrong as semantics, and both are fixed *in the record only*.
    Keeping the verdict and the joined detail string byte-identical is what
    guarantees the harness's catch_rate / false_positive_rate /
    missed_mutants outputs are unaffected by adding instrumentation, so the
    re-run stays comparable to previous runs. Correcting the verdict here
    would silently change the benchmark's headline numbers at the same time
    as adding attribution, and the two effects would be inseparable. That
    fix is tracked as its own item (see checker.py:229, same coercion).

    `record["outcome"]` carries the true four-valued result:

      pass  -- check ran and agreed with the reference
      fail  -- real numeric/property disagreement (a genuine catch)
      error -- the check raised; attribution is SUSPECT, not a catch
      skip  -- the check declined to run (returned None)

    The error/fail distinction is the whole point. In the AutoKernel
    baseline audit (benchmarks/autokernel/AUTOKERNEL_BASELINE_AUDIT.md) an
    identical bare-except pattern scored a ValueError raised by a
    wrongly-built input as a legitimate gate failure, which produced that
    baseline's entire reported 18% false-positive rate. The same hazard
    exists here: shape_generalization.py's monotone_rows variant used to
    raise RuntimeError on 1-D primaries and, in its own words,
    "coincidentally match[ed] the expected mutant-catch verdict". Folding
    errors into catches would make a crashing check look like a working one
    in exactly the ablation table meant to find dead checks.

    `record["subchecks"]` holds per-sub-check outcomes for compound checks
    (cross_shape's 5 shapes, weight_magnitude's 4 variants) when the check
    function supplies a 3rd return element THAT IS A LIST, else None.

    The isinstance guard is load-bearing, not defensive padding. A 3rd
    element is only meaningful here as a list of sub-check records; any
    other type means the check is using that slot for something else and
    is not compound. tile_coverage.py's softmax-positivity check used to
    return an int column count there, which this function copied verbatim
    into the record. It surfaced in only 2 of 322 records in a full corpus
    run, and crashed benchmarks/analyze_check_ablation.py with
    `TypeError: 'int' object is not iterable` -- the ablation table could
    not be built at all from real GPU data. Dropping a non-list keeps one
    misbehaving check from corrupting the ablation input for every other
    check. Permanent negative control:
    tests/instrumentation/check_ablation_report.py.
    """
    subchecks = None
    # Timer brackets fn() ONLY -- not the bool coercion, not record
    # construction. perf_counter costs ~50ns against ms-scale checks, so the
    # instrumentation is under 0.01% of what it measures.
    _t0 = time.perf_counter()
    try:
        result = fn()
        _elapsed_ms = 1000.0 * (time.perf_counter() - _t0)
        if isinstance(result, (list, tuple)):
            raw = result[0]
            detail = str(result[1]) if len(result) > 1 else None
            if len(result) > 2 and isinstance(result[2], list):
                subchecks = result[2]
        else:
            raw = result
            detail = None
        passed = bool(raw)                      # legacy coercion, unchanged
        if raw is None:
            outcome = "skip"
        elif passed:
            outcome = "pass"
        else:
            outcome = "fail"
    except Exception as e:
        # Time the failure too. A check that raises after 3s is a very
        # different cost profile from one that raises immediately, and the
        # ablation cannot see the difference if errors are recorded as
        # untimed.
        _elapsed_ms = 1000.0 * (time.perf_counter() - _t0)
        detail = f"{type(e).__name__}: {e}"
        passed = False                          # legacy behaviour, unchanged
        outcome = "error"

    record = {"name": name, "outcome": outcome,
              "detail": detail, "subchecks": subchecks,
              "duration_ms": _elapsed_ms}
    return name, passed, detail, record


def _record(name, passed, detail, outcome):
    """Build a check tuple for a check that was resolved without _try
    (e.g. adversarial-input generation failing before any check ran).

    `duration_ms` is None, not 0.0. These checks never executed, and "never
    ran" must stay distinguishable from "ran instantly" -- the same reason the
    outcome field is four-valued rather than a bool. A 0.0 here would silently
    pull the per-check mean toward zero and make a check that was skipped look
    like the fastest check in the suite.
    """
    return name, passed, detail, {"name": name, "outcome": outcome,
                                  "detail": detail, "subchecks": None,
                                  "duration_ms": None}


def _summarize(checks):
    """Returns (passed, detail_string, records).

    `passed` and `detail_string` are computed exactly as before this
    instrumentation existed -- same predicate (`not c[1]`), same join, same
    order -- so downstream harness output is bit-for-bit unchanged. The
    third element is purely additive.
    """
    failed = [c for c in checks if not c[1]]
    records = [c[3] for c in checks]
    if failed:
        return False, "; ".join(f"{n}: {d}" for n, _, d, _r in failed), records
    return True, None, records


def _cand_ref_wrappers(spec, candidate_fn, reference_fn, inputs):
    def _cand(x):
        new_inputs = (x,) + inputs[1:] if isinstance(inputs, tuple) else x
        return spec.run_candidate(candidate_fn, new_inputs)

    def _ref(x):
        new_inputs = (x,) + inputs[1:] if isinstance(inputs, tuple) else x
        return spec.run_reference(reference_fn, new_inputs)

    return _cand, _ref


def _run_structural(spec, candidate_fn, raw_kernel, reference_fn, inputs):
    """Structural layer (Layer 1). Named by WHAT it checks, not by its
    position: the layer ORDER changed on 2026-08-20 (numeric moved last),
    and an identifier that encodes position goes stale the moment that
    happens. The ablation systems run one layer each, unconditionally, so
    they are unaffected by the reorder."""
    primary = spec.primary_input(inputs)
    _cand, _ref = _cand_ref_wrappers(spec, candidate_fn, reference_fn, inputs)

    checks = [
        _try("nan_inf", lambda: check_nan_inf(_cand, primary)),
        _try("dtype_preserved",
             lambda: check_dtype_preserved(_cand, primary, expected_dtype=spec.output_dtype)),
        _try("ghost_optimization", lambda: check_ghost_optimization(candidate_fn)),
        _try("timing_manipulation", lambda: check_timing_manipulation(candidate_fn)),
        _try("partial_computation", lambda: check_partial_computation(candidate_fn)),
        _try("determinism", lambda: check_determinism(_cand, primary)),
        # spec/inputs/raw_* enable the companion-perturbation rung -- see the
        # matching call site in verification/checker.py. Kept in step with it
        # deliberately: this adapter exists to run the SAME checks the real
        # checker runs, so a divergence here silently benchmarks a different
        # checker than the one being shipped.
        _try("kernel_executed",
             lambda: check_kernel_executed(
                 _cand, primary, _ref,
                 spec=spec, inputs=inputs,
                 raw_candidate_fn=candidate_fn, raw_reference_fn=reference_fn)),
        _try("tile_coverage_structural",
             lambda: check_all_tiles_visited_generic(spec, candidate_fn, inputs)),
    ]
    if primary.dim() == 2 and not isinstance(inputs, tuple) and spec.name == "softmax":
        checks.append(_try("tile_coverage_softmax_positivity",
                            lambda: check_all_tiles_visited(candidate_fn, raw_kernel, primary)))

    return _summarize(checks)


def _run_numeric(spec, candidate_fn, reference_fn, inputs):
    primary = spec.primary_input(inputs)
    _cand, _ref = _cand_ref_wrappers(spec, candidate_fn, reference_fn, inputs)

    checks = [
        _try("output_shape", lambda: check_output_shape(_cand, _ref, primary)),
        _try("perturbation_tolerance", lambda: check_perturbation_tolerance(_cand, _ref, primary)),
        _try("cross_shape", lambda: _check_cross_shape(candidate_fn, reference_fn, spec)),
        _try("weight_magnitude", lambda: check_weight_magnitude(candidate_fn, reference_fn, spec)),
    ]
    if spec.requires_backward:
        checks.append(_try("backward_pass", lambda: check_backward_pass(_cand, _ref, primary)))

    # Adversarial inputs -- part of Layer 2, not an optional extra.
    try:
        adversarial_pairs = spec.get_adversarial_inputs(inputs)
    except Exception as e:
        adversarial_pairs = []
        checks.append(_record("adversarial_setup", False,
                              f"Could not generate adversarial inputs: {e}", "error"))

    for name, adv_inputs in adversarial_pairs:
        adv_primary = spec.primary_input(adv_inputs)

        def _adv_cand(x, ai=adv_inputs):
            new_inputs = (x,) + ai[1:] if isinstance(ai, tuple) else x
            return spec.run_candidate(candidate_fn, new_inputs)

        def _adv_ref(x, ai=adv_inputs):
            new_inputs = (x,) + ai[1:] if isinstance(ai, tuple) else x
            return spec.run_reference(reference_fn, new_inputs)

        if spec.output_dtype is not None:
            checks.append(_try(f"adversarial_{name}",
                                lambda c=_adv_cand, r=_adv_ref, ap=adv_primary:
                                    _check_exact_match(c, r, ap)))
        else:
            checks.append(_try(f"adversarial_{name}",
                                lambda c=_adv_cand, r=_adv_ref, ap=adv_primary:
                                    check_perturbation_tolerance(c, r, ap)))

    return _summarize(checks)


def _run_algebraic(spec, candidate_fn, inputs):
    checks = [
        _try(prop_name, lambda fn=prop_fn: fn(candidate_fn, inputs))
        for prop_name, prop_fn in spec.algebraic_properties
    ]
    return _summarize(checks)


def _get_torch_inputs(entry, rng):
    np_args = entry["input_fn"](rng)
    return entry["to_torch"](np_args)


def my_checker_system(entry, is_mutant, rng):
    """Full three-layer checker (verification.checker.KernelChecker)."""
    spec = entry["spec"]
    candidate_fn = entry["torch_mutant_fn"] if is_mutant else entry["torch_ref_fn"]
    raw_kernel = entry["raw_kernel_mutant"] if is_mutant else entry["raw_kernel_ref"]
    reference_fn = entry["torch_ref_fn"]

    # TIMER SCOPE: t0 before _get_torch_inputs -- input generation is part
    # of what a checking system costs. See run()'s convention note.
    t0 = time.perf_counter()
    inputs = _get_torch_inputs(entry, rng)
    checker = KernelChecker(spec)
    results = checker.run(candidate_fn, raw_kernel, reference_fn, inputs)
    dt = time.perf_counter() - t0

    passed = all(r.passed for r in results)
    detail = None
    if not passed:
        detail = "; ".join(f"[L{r.layer}]{r.check_name}" for r in results if not r.passed)

    # Records for the FULL checker are emitted too, but they are NOT a valid
    # per-check ablation: KernelChecker.run short-circuits between layers
    # (checker.py:114/155/213), so a check only appears here if every earlier
    # layer passed. Marked short_circuited=True so the analysis script uses
    # the three single-layer ablations for attribution and these only for
    # "which layer fired first in the real pipeline" (item #3).
    #
    # CheckResult carries no skip/error distinction -- KernelChecker's own
    # _run_check does the same bool() coercion -- so outcome is reported as
    # the coarse pass/fail it actually is, not guessed at.
    # `subchecks` and `duration_ms` are ADDITIVE fields, both None unless
    # KCC_CHECK_TIMING=1. subchecks was previously hardcoded None here, which
    # is why check_weight_magnitude's four per-variant outcomes -- which it has
    # always returned as a third element -- never reached any artifact.
    # `scope_flags` is the third additive field, None unless
    # KCC_SCOPE_DETECT=1. Serialised HERE rather than left to be dug out of
    # `subchecks`: the promotion in KernelChecker._run_check is the thing the
    # GPU arm has to validate, and reading the record from `subchecks` would
    # exercise the wrong half and pass whether or not the promotion works.
    records = [{"name": r.check_name, "layer": r.layer,
                "outcome": "pass" if r.passed else "fail",
                "detail": r.details, "subchecks": r.subchecks,
                "duration_ms": r.duration_ms,
                "scope_flags": r.scope_flags,
                "short_circuited": True}
               for r in results]
    return passed, dt, detail, records


def my_checker_structural_only(entry, is_mutant, rng):
    spec = entry["spec"]
    candidate_fn = entry["torch_mutant_fn"] if is_mutant else entry["torch_ref_fn"]
    raw_kernel = entry["raw_kernel_mutant"] if is_mutant else entry["raw_kernel_ref"]
    reference_fn = entry["torch_ref_fn"]

    # TIMER SCOPE: t0 before _get_torch_inputs -- input generation is part
    # of what a checking system costs. See run()'s convention note.
    t0 = time.perf_counter()
    inputs = _get_torch_inputs(entry, rng)
    passed, detail, records = _run_structural(spec, candidate_fn, raw_kernel, reference_fn, inputs)
    dt = time.perf_counter() - t0
    return passed, dt, detail, records


def my_checker_numeric_only(entry, is_mutant, rng):
    spec = entry["spec"]
    candidate_fn = entry["torch_mutant_fn"] if is_mutant else entry["torch_ref_fn"]
    reference_fn = entry["torch_ref_fn"]

    # TIMER SCOPE: t0 before _get_torch_inputs -- input generation is part
    # of what a checking system costs. See run()'s convention note.
    t0 = time.perf_counter()
    inputs = _get_torch_inputs(entry, rng)
    passed, detail, records = _run_numeric(spec, candidate_fn, reference_fn, inputs)
    dt = time.perf_counter() - t0
    return passed, dt, detail, records


def my_checker_algebraic_only(entry, is_mutant, rng):
    spec = entry["spec"]
    candidate_fn = entry["torch_mutant_fn"] if is_mutant else entry["torch_ref_fn"]

    # TIMER SCOPE: t0 before _get_torch_inputs -- input generation is part
    # of what a checking system costs. See run()'s convention note.
    t0 = time.perf_counter()
    inputs = _get_torch_inputs(entry, rng)
    passed, detail, records = _run_algebraic(spec, candidate_fn, inputs)
    dt = time.perf_counter() - t0
    return passed, dt, detail, records
