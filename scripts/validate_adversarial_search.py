"""
scripts/validate_adversarial_search.py

Integration test suite for the adversarial search pipeline.

These are the pipeline's positive controls — must all pass before any
results from the search loop can be trusted in a paper context.

Tests:
  1.  schema_roundtrip         InputProposal → dict → InputProposal lossless
  2.  materializer_shape       TensorDescriptor → correct tensor shape
  3.  materializer_patch       Patch applied to correct slice
  4.  materializer_scale_shift scale and shift applied correctly
  5.  validate_missing_key     Schema rejects missing tensor keys
  6.  validate_bad_fill        Schema rejects invalid fill strategy
  7.  verdict_hit_logic        HIT requires ref pass + mutant fail + gap
  8.  verdict_miss_ref_fails   is_hit=False when reference fails checker
  9.  verdict_no_gap           is_hit=False when mutant fails naive too
  10. strategy_greedy          GreedyStrategy selects single best
  11. strategy_beam            BeamSearchStrategy selects top-B
  12. strategy_diverse         DiverseBeamStrategy diversifies patterns
  13. history_store            SQLite store: create/save/resume round-trip
  14. end_to_end_smoke         One worker, one iteration (skipped without key)

Usage:
    python scripts/validate_adversarial_search.py
    python scripts/validate_adversarial_search.py --skip-llm
    python scripts/validate_adversarial_search.py --model gpt-4o
"""

import argparse
import os
import sys
import tempfile
import traceback
import types
import uuid
from pathlib import Path

CHECKER_ROOT = os.environ.get("CHECKER_ROOT", str(Path(__file__).parent.parent))
if CHECKER_ROOT not in sys.path:
    sys.path.insert(0, CHECKER_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch

from verification.adversarial_search.schemas import (
    InputProposal, TensorDescriptor, ProposalVerdict,
    KernelExecutionResult, ExecutionError, SearchResult,
    validate_proposal,
)
from verification.adversarial_search.materializer import (
    materialize_proposal, tensors_to_inputs,
)
from verification.adversarial_search.strategy import (
    GreedyStrategy, BeamSearchStrategy, DiverseBeamStrategy,
)
from verification.adversarial_search.history.store import SearchHistoryStore
from verification.adversarial_search.coordinator import SearchCoordinator


PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_proposal(
    operator="softmax",
    shape=(4, 16),
    fill="randn",
    patches=None,
    scale=1.0,
    shift=0.0,
    pattern="partial_tile",
) -> InputProposal:
    required = {
        "softmax":         {"x": (shape, fill, scale, shift, patches or [])},
        "layernorm":       {"x": (shape, fill, scale, shift, []),
                            "gamma": ([shape[-1]], "ones", 1.0, 0.0, []),
                            "beta":  ([shape[-1]], "zeros", 1.0, 0.0, [])},
        "rmsnorm":         {"x": (shape, fill, scale, shift, []),
                            "gamma": ([shape[-1]], "ones", 1.0, 0.0, [])},
    }
    tensors = {}
    spec = required.get(operator, {"x": (shape, fill, scale, shift, patches or [])})
    for name, (sh, fi, sc, sh2, pa) in spec.items():
        tensors[name] = TensorDescriptor(
            shape=list(sh), dtype="float32", fill=fi,
            scale=sc, shift=sh2, patches=pa,
        )
    return InputProposal(
        proposal_id=str(uuid.uuid4()),
        worker_id="test",
        iteration=0,
        operator=operator,
        tensors=tensors,
        rationale="test",
        predicted_failure_mode=pattern,
    )


def _make_execution_result(
    proposal_id, kernel_id,
    passed_checker=True, passed_naive=True, error=None,
) -> KernelExecutionResult:
    return KernelExecutionResult(
        proposal_id=proposal_id,
        kernel_id=kernel_id,
        passed_checker=passed_checker,
        passed_naive=passed_naive,
        error=error,
        check_results=[],
        wall_time_ms=1.0,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_schema_roundtrip():
    original = _make_proposal(patches=[{"indices": "[:, -1]", "value": 1e9}])
    recovered = InputProposal.from_dict(original.to_dict())
    assert recovered.proposal_id == original.proposal_id
    assert recovered.tensors["x"].patches == original.tensors["x"].patches
    return PASS, "lossless round-trip"


def test_materializer_shape():
    p = _make_proposal(shape=(8, 33))
    t = materialize_proposal(p, device="cpu")["x"]
    assert t.shape == (8, 33), f"Expected (8,33), got {t.shape}"
    return PASS, f"shape={tuple(t.shape)}"


def test_materializer_patch():
    p = _make_proposal(patches=[{"indices": "[:, -1]", "value": 1e9}])
    t = materialize_proposal(p, device="cpu")["x"]
    assert (t[:, -1] == 1e9).all(), "patch not applied"
    assert not (t[:, :-1] == 1e9).all(), "patch leaked"
    return PASS, "patch applied to [:, -1] only"


def test_materializer_scale_shift():
    p = _make_proposal(fill="ones", scale=3.0, shift=1.0)
    t = materialize_proposal(p, device="cpu")["x"]
    # ones * 3.0 + 1.0 = 4.0
    assert torch.allclose(t, torch.full_like(t, 4.0)), f"Expected 4.0, got {t.unique()}"
    return PASS, "scale=3.0, shift=1.0 → value=4.0"


def test_validate_missing_key():
    p = _make_proposal(operator="layernorm")
    # Remove gamma
    del p.tensors["gamma"]
    ok, msg = validate_proposal(p)
    assert not ok and "gamma" in msg
    return PASS, f"correctly rejected: {msg}"


def test_validate_bad_fill():
    p = _make_proposal()
    p.tensors["x"].fill = "gaussian"
    ok, msg = validate_proposal(p)
    assert not ok and "fill" in msg
    return PASS, f"correctly rejected: {msg}"


def test_verdict_hit_logic():
    coord = types.SimpleNamespace()
    coord._evaluate_verdict = SearchCoordinator._evaluate_verdict.__get__(
        coord, SearchCoordinator
    )
    p = _make_proposal()
    ref  = _make_execution_result(p.proposal_id, "reference", passed_checker=True,  passed_naive=True)
    ma   = _make_execution_result(p.proposal_id, "mutant_a",  passed_checker=False, passed_naive=True)  # gap!
    mb   = _make_execution_result(p.proposal_id, "mutant_b",  passed_checker=False, passed_naive=False) # naive also catches — not interesting
    v = coord._evaluate_verdict(p, ref, [ma, mb])
    assert v.is_hit
    assert "mutant_a" in v.hit_mutants
    assert "mutant_b" in v.missed_mutants
    assert v.gap_confirmed
    return PASS, "HIT logic correct"


def test_verdict_miss_ref_fails():
    coord = types.SimpleNamespace()
    coord._evaluate_verdict = SearchCoordinator._evaluate_verdict.__get__(
        coord, SearchCoordinator
    )
    p = _make_proposal()
    ref = _make_execution_result(p.proposal_id, "reference", passed_checker=False)
    m   = _make_execution_result(p.proposal_id, "mutant_a",  passed_checker=False, passed_naive=True)
    v = coord._evaluate_verdict(p, ref, [m])
    assert not v.is_hit
    assert not v.reference_passed
    return PASS, "correctly rejected when reference fails"


def test_verdict_no_gap():
    """is_hit must be False when mutant fails both checker AND naive (naive would have caught it)."""
    coord = types.SimpleNamespace()
    coord._evaluate_verdict = SearchCoordinator._evaluate_verdict.__get__(
        coord, SearchCoordinator
    )
    p = _make_proposal()
    ref = _make_execution_result(p.proposal_id, "reference", passed_checker=True)
    m   = _make_execution_result(p.proposal_id, "mutant_a",  passed_checker=False, passed_naive=False)
    v = coord._evaluate_verdict(p, ref, [m])
    assert not v.is_hit
    assert not v.gap_confirmed
    return PASS, "correctly rejected when no gap (naive also catches)"


def test_strategy_greedy():
    s = GreedyStrategy()
    pairs = []
    for i, score in enumerate([5.0, 15.0, 3.0]):
        p = _make_proposal()
        v = ProposalVerdict(p.proposal_id, False, [], [], True, False, "", beam_score=score)
        pairs.append((p, v))
    selected = s.select(pairs, beam_width=4)
    assert len(selected) == 1
    assert selected[0][1].beam_score == 15.0
    return PASS, "greedy selects single best"


def test_strategy_beam():
    s = BeamSearchStrategy()
    scores = [1.0, 9.0, 7.0, 3.0, 5.0]
    pairs = []
    for score in scores:
        p = _make_proposal()
        v = ProposalVerdict(p.proposal_id, False, [], [], True, False, "", beam_score=score)
        pairs.append((p, v))
    selected = s.select(pairs, beam_width=3)
    assert len(selected) == 3
    sel_scores = sorted([x[1].beam_score for x in selected], reverse=True)
    assert sel_scores == [9.0, 7.0, 5.0]
    return PASS, "beam selects top-3"


def test_strategy_diverse():
    s = DiverseBeamStrategy(diversity_weight=10.0)
    patterns = ["partial_tile", "partial_tile", "wrong_reduction", "boundary_mask"]
    scores   = [10.0, 8.0, 7.0, 6.0]
    pairs = []
    for pattern, score in zip(patterns, scores):
        p = _make_proposal(pattern=pattern)
        v = ProposalVerdict(p.proposal_id, False, [], [], True, False, "", beam_score=score)
        pairs.append((p, v))
    selected = s.select(pairs, beam_width=3)
    sel_patterns = [x[0].predicted_failure_mode for x in selected]
    # High diversity weight should penalise the second partial_tile
    assert "wrong_reduction" in sel_patterns or "boundary_mask" in sel_patterns, \
        f"Diversity failed: got {sel_patterns}"
    return PASS, f"diverse beam patterns: {sel_patterns}"


def test_history_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with SearchHistoryStore(db_path) as store:
            run_id = store.create_run("r1", "softmax", "beam", "test-model", 4, 20)
            p = _make_proposal()
            store.save_proposal("r1", p)
            v = ProposalVerdict(p.proposal_id, True, ["m1"], [], True, True, "hit", beam_score=12.0)
            store.save_verdict("r1", v)
            # Resume
            ctx = store.resume_run("r1")
            assert ctx is not None
            assert ctx["n_proposals"] == 1
            # Coverage
            store.add_memory_item("softmax", "partial_tile", "summary text", "r1")
            items = store.get_memory_items("softmax", limit=5)
            assert len(items) == 1 and items[0]["bug_pattern"] == "partial_tile"
    return PASS, "SQLite store: create/save/resume/memory round-trip"


def test_end_to_end_smoke(model: str):
    ref_path = os.path.join(CHECKER_ROOT, "TritonBench/reference/softmax.py")
    mutant_path = os.path.join(CHECKER_ROOT, "TritonBench/cheating/softmax/first_tile.py")
    if not os.path.exists(ref_path) or not os.path.exists(mutant_path):
        return SKIP, "TritonBench kernels not found"

    with tempfile.TemporaryDirectory() as tmpdir:
        coord = SearchCoordinator(
            operator="softmax",
            reference_src_path=ref_path,
            mutant_src_paths={"first_tile": mutant_path},
            model=model,
            strategy="greedy",
            n_workers=1,
            max_iterations=1,
            timeout_per_exec=30,
            output_dir=tmpdir,
        )
        result = coord.run()
        assert result.total_proposals >= 1
    return PASS, f"pipeline completed: proposals={result.total_proposals}, hit={result.winning_proposal is not None}"


# ── Runner ────────────────────────────────────────────────────────────────────

UNIT_TESTS = [
    ("schema_roundtrip",       test_schema_roundtrip),
    ("materializer_shape",     test_materializer_shape),
    ("materializer_patch",     test_materializer_patch),
    ("materializer_scale_shift", test_materializer_scale_shift),
    ("validate_missing_key",   test_validate_missing_key),
    ("validate_bad_fill",      test_validate_bad_fill),
    ("verdict_hit_logic",      test_verdict_hit_logic),
    ("verdict_miss_ref_fails", test_verdict_miss_ref_fails),
    ("verdict_no_gap",         test_verdict_no_gap),
    ("strategy_greedy",        test_strategy_greedy),
    ("strategy_beam",          test_strategy_beam),
    ("strategy_diverse",       test_strategy_diverse),
    ("history_store",          test_history_store),
]


def run_all(skip_llm: bool, model: str):
    results = []
    for name, fn in UNIT_TESTS:
        try:
            status, detail = fn()
        except Exception as e:
            status = FAIL
            detail = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        results.append((name, status, detail))

    if not skip_llm:
        try:
            status, detail = test_end_to_end_smoke(model)
        except Exception as e:
            status = FAIL
            detail = f"{type(e).__name__}: {e}"
        results.append(("end_to_end_smoke", status, detail))
    else:
        results.append(("end_to_end_smoke", SKIP, "skipped via --skip-llm"))

    width = max(len(n) for n, _, _ in results) + 2
    print(f"\n{'='*65}\n  Adversarial Search Pipeline — Validation\n{'='*65}")
    for name, status, detail in results:
        print(f"  {status:<4}  {name:<{width}}  {detail}")

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    print(f"\n  {n_pass} passed  {n_fail} failed  {n_skip} skipped\n")

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()
    run_all(skip_llm=args.skip_llm, model=args.model)