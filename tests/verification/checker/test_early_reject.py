"""
Negative control for the layer short-circuit ("early reject") in
KernelChecker.run.

WHAT THIS GUARDS
----------------
`KernelChecker.run` short-circuits between layers (verification/checker.py:114,
163, 171): a failure in structural (Layer 1) returns immediately without ever
running algebraic (Layer 2) or the expensive numeric oracle (Layer 3).

The gate is a single condition per layer, and it is invertible in a way that is
silent and catastrophic:

    if any(not r.passed for r in results):   # correct: bail out ON FAILURE
        return results

Flip the sense of that test -- `if all(r.passed ...)` -- and the checker keeps
returning verdicts, keeps passing every other test in this suite, and quietly
stops verifying anything: candidates that PASS structural would return early and
never be checked numerically at all. A mutation that turns a verification tool
into a rubber stamp must not be able to hide behind a green suite.

`tests/instrumentation/check_layer_order.py` already asserts the ORDER of the
layers and the presence of three gates, but it does so by reading checker.py as
TEXT. It never executes the pipeline, so it cannot observe the DIRECTION of the
short-circuit, nor whether a gate that is present actually fires. These tests
execute `KernelChecker.run` end-to-end and assert on the records it really
produced -- which layers ran, whether the algebraic properties were invoked at
all, and what verdict came out.

WHY THE SPEC DECLARES A REAL ALGEBRAIC PROPERTY
-----------------------------------------------
An earlier draft of this file used a spec with NO algebraic properties, which
made its "Layer 2 did not run" assertion vacuous -- Layer 2 could never appear
in the records whether the gate fired or not. Mutation-tested and confirmed:
DELETING the Layer-1 gate outright left all six tests green, because the Layer-2
gate downstream caught the same structural failure and still skipped numeric.
The observable verdict was identical, so the test could not tell a present gate
from an absent one.

`_rows_sum_to_one` fixes that. It is a genuine softmax invariant, and it records
each invocation in `_ALGEBRAIC_CALLS`, which gives the tests a direct probe for
"did Layer 2 actually execute" rather than inferring it from the record set.
Removing the Layer-1 gate now shows up as an algebraic property that ran when it
should not have.

WHY THE TEST DOUBLES LOOK LIKE THIS
-----------------------------------
These tests must run on CPU-only machines (the whole suite does; `triton` is a
linux-only dependency per pyproject.toml and is not installed on darwin).

That creates one obstacle. Two Layer-1 structural checks -- `ghost_optimization`
and `partial_computation` -- are pure AST analyses that look for a Triton launch
in the candidate's SOURCE TEXT. A plain-PyTorch stand-in therefore fails both
unconditionally ("No Triton kernel launch detected", "delegation ratio 100%"),
which would make *every* candidate here a structural failure and leave the
pass-side direction untested -- exactly the half that matters most.

`_FakeJIT` resolves that: it supports the `kernel[grid](...)` subscript-call
syntax, so the candidate's source parses as a genuine Triton launch, while the
call itself executes plain PyTorch on CPU. The structural AST checks are purely
syntactic, so this is a faithful stand-in for them and not a way around them --
verified in-test by `test_double_is_structurally_clean`, which fails loudly if a
future change to the AST analysis stops accepting this shape. Without that
canary, a stricter analyser would silently turn the pass-side tests into vacuous
ones (everything fails structurally, so "skipped numeric" is trivially true and
proves nothing).
"""
import pytest
import torch

from verification.checker import KernelChecker
from verification.layer1_structural.ast_analysis import (
    check_ghost_optimization,
    check_partial_computation,
    check_timing_manipulation,
)
from verification.specs.base_spec import SingleTensorSpec

pytestmark = pytest.mark.checker


# ── Test doubles ─────────────────────────────────────────────────────────────

class _FakeJIT:
    """
    Stands in for a `@triton.jit` kernel: supports the `kernel[grid](args)`
    launch syntax the Layer-1 AST checks scan for, but runs on CPU.
    """

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, grid):
        return self._fn


def _softmax_body(out, x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    out.copy_(e / e.sum(dim=-1, keepdim=True))


def _hot_softmax_body(out, x):
    # A real softmax of 2x: still a valid probability distribution per row, so
    # it SATISFIES the algebraic invariant and can only be caught numerically.
    m = (2 * x).max(dim=-1, keepdim=True).values
    e = torch.exp(2 * x - m)
    out.copy_(e / e.sum(dim=-1, keepdim=True))


def _column_normalised_body(out, x):
    # Normalises down the WRONG axis: rows no longer sum to 1, so the algebraic
    # property catches it before numeric ever runs.
    e = torch.exp(x - x.max())
    out.copy_(e / e.sum(dim=0, keepdim=True))


softmax_kernel = _FakeJIT(_softmax_body)
hot_softmax_kernel = _FakeJIT(_hot_softmax_body)
column_softmax_kernel = _FakeJIT(_column_normalised_body)


def correct_candidate(x):
    out = torch.empty_like(x)
    grid = (x.shape[0],)
    softmax_kernel[grid](out, x)
    return out


def numerically_wrong_candidate(x):
    """
    Structurally clean AND algebraically clean, but numerically wrong.
    Must reach Layer 3 and fail there.
    """
    out = torch.empty_like(x)
    grid = (x.shape[0],)
    hot_softmax_kernel[grid](out, x)
    return out


def algebraically_wrong_candidate(x):
    """Structurally clean, violates the row-sum invariant. Must stop at Layer 2."""
    out = torch.empty_like(x)
    grid = (x.shape[0],)
    column_softmax_kernel[grid](out, x)
    return out


def structurally_bad_candidate(x):
    """
    Fails Layer 1 (`timing_manipulation`: samples a timer, never synchronises)
    while computing the CORRECT answer.

    The correctness is the whole point of this double. If the early-reject were
    removed or inverted, this candidate would sail through the numeric layer --
    it agrees with the reference to the bit -- so the ONLY thing that can
    produce a FAIL verdict here is the structural layer actually gating the
    result. That makes it a true negative control rather than a candidate that
    would have failed anyway for unrelated reasons.
    """
    import time
    t0 = time.perf_counter()
    out = torch.empty_like(x)
    grid = (x.shape[0],)
    softmax_kernel[grid](out, x)
    _elapsed = time.perf_counter() - t0
    return out


def reference_softmax(x):
    return torch.softmax(x, dim=-1)


# ── Spec ─────────────────────────────────────────────────────────────────────

# Records every invocation of the algebraic property, so a test can assert that
# Layer 2 did not merely fail to produce records -- it never ran at all.
_ALGEBRAIC_CALLS = []


def _rows_sum_to_one(candidate_fn, inputs):
    _ALGEBRAIC_CALLS.append(inputs)
    out = candidate_fn(inputs)
    sums = out.sum(dim=-1)
    dev = (sums - 1.0).abs().max().item()
    return dev < 1e-5, f"max row-sum deviation {dev:.3e}"


class _CPUSoftmaxSpec(SingleTensorSpec):
    """Minimal CPU-runnable spec: one algebraic property, no backward pass."""

    def get_adversarial_inputs(self, inputs):
        return []

    @property
    def algebraic_properties(self):
        return [("rows_sum_to_one", _rows_sum_to_one)]

    @property
    def valid_shapes(self):
        return [(32, 16), (8, 64)]


def _spec():
    return _CPUSoftmaxSpec(name="softmax", requires_backward=False)


def _run(candidate):
    _ALGEBRAIC_CALLS.clear()
    checker = KernelChecker(_spec())
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    results = checker.run(candidate, softmax_kernel, reference_softmax, x)
    return checker, results


def _layers_run(results):
    return {r.layer for r in results}


# ── Canary: the doubles must remain structurally clean ───────────────────────

def test_double_is_structurally_clean():
    """
    Guards the other tests from becoming vacuous.

    If a future change to the AST analysis stops recognising `_FakeJIT`'s launch
    syntax, `correct_candidate` starts failing Layer 1, and every "numeric layer
    ran" assertion below would silently invert its meaning.
    """
    for check in (check_ghost_optimization, check_partial_computation,
                  check_timing_manipulation):
        passed, detail = check(correct_candidate)[:2]
        assert passed, f"{check.__name__} rejected the test double: {detail}"


# ── The negative control ─────────────────────────────────────────────────────

def test_structural_failure_returns_fail_not_pass():
    """
    THE negative control: a structurally-failing kernel must return a FAIL
    verdict via early-reject, never an incorrect PASS.

    This candidate is numerically CORRECT, so a broken gate cannot be rescued
    by a downstream layer noticing the bug -- there is no numeric bug to notice.
    """
    checker, results = _run(structurally_bad_candidate)
    verdict = checker.verdict(results)

    assert verdict.startswith("FAIL"), (
        f"structurally-failing kernel returned {verdict!r} -- early-reject let "
        "a bad kernel through"
    )
    failed = [r for r in results if not r.passed]
    assert any(r.layer == 1 for r in failed), (
        f"expected a Layer-1 failure, got {[(r.layer, r.check_name) for r in failed]}"
    )
    assert any(r.check_name == "timing_manipulation" and not r.passed
               for r in results), "expected timing_manipulation to be the catch"


def test_structural_failure_skips_algebraic_and_numeric():
    """
    The perf claim, and the guard against the Layer-1 gate being deleted.

    `_ALGEBRAIC_CALLS` is the load-bearing assertion: absence of Layer-2 records
    alone would NOT detect a deleted Layer-1 gate, because the Layer-2 gate
    downstream catches the same failure and still skips numeric. Only observing
    that the algebraic property never executed distinguishes the two.
    """
    _, results = _run(structurally_bad_candidate)

    assert _ALGEBRAIC_CALLS == [], (
        f"algebraic property ran {len(_ALGEBRAIC_CALLS)} time(s) despite a "
        "structural failure -- the Layer-1 gate did not fire"
    )
    assert 2 not in _layers_run(results), (
        f"algebraic layer produced records despite a structural failure; "
        f"layers seen: {sorted(_layers_run(results))}"
    )
    assert 3 not in _layers_run(results), (
        "numeric layer ran despite a structural failure -- early-reject did "
        f"not fire; layers seen: {sorted(_layers_run(results))}"
    )


def test_early_reject_runs_strictly_fewer_checks():
    """Early-reject must actually shorten the run, not just relabel it."""
    _, rejected = _run(structurally_bad_candidate)
    _, full = _run(correct_candidate)

    assert len(rejected) < len(full), (
        f"early-rejected run executed {len(rejected)} checks vs {len(full)} for "
        "a full run -- no work was skipped"
    )


def test_algebraic_failure_skips_numeric():
    """The second gate: an algebraic failure must not pay for the numeric layer."""
    checker, results = _run(algebraically_wrong_candidate)
    verdict = checker.verdict(results)

    assert verdict.startswith("FAIL"), f"algebraically wrong kernel returned {verdict!r}"
    assert 2 in _layers_run(results), "algebraic layer never ran"
    assert 3 not in _layers_run(results), (
        "numeric layer ran despite an algebraic failure; layers seen: "
        f"{sorted(_layers_run(results))}"
    )


# ── The inversion guard (the direction that is easy to get backwards) ────────

def test_structural_pass_does_not_skip_numeric():
    """
    The inverted-condition guard. A structurally AND algebraically clean but
    numerically WRONG kernel must reach Layer 3 and fail there.

    If either gate were flipped to bail out on success, this candidate would
    return early with an all-passed record set and be reported PASS.
    """
    checker, results = _run(numerically_wrong_candidate)
    verdict = checker.verdict(results)

    assert 3 in _layers_run(results), (
        "numeric layer was skipped for a kernel that passed the cheap layers -- "
        f"the short-circuit is inverted; layers seen: {sorted(_layers_run(results))}"
    )
    assert verdict.startswith("FAIL"), f"numerically wrong kernel returned {verdict!r}"
    assert all(r.passed for r in results if r.layer < 3), (
        "this candidate was supposed to clear Layers 1 and 2; the test is not "
        "exercising the pass-side path"
    )


def test_correct_kernel_runs_every_layer_and_passes():
    """Control: a correct kernel is not short-circuited anywhere."""
    checker, results = _run(correct_candidate)
    verdict = checker.verdict(results)

    assert verdict == "PASS", f"correct kernel returned {verdict!r}"
    assert _layers_run(results) == {1, 2, 3}, (
        f"expected all three layers to run; saw {sorted(_layers_run(results))}"
    )
    assert len(_ALGEBRAIC_CALLS) == 1, (
        f"algebraic property ran {len(_ALGEBRAIC_CALLS)} time(s), expected 1"
    )
