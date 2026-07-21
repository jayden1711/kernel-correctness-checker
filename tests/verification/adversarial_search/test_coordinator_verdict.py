"""
tests/verification/adversarial_search/test_coordinator_verdict.py

Tests for the coordinator's verdict evaluation logic.

The hit invariant is the paper's central claim:
  1. reference passes full checker    (input is semantically valid)
  2. at least one mutant fails checker (bug is exposed)
  3. that mutant ALSO passed naive allclose (gap confirmed —
     the bug is invisible to naive testing)

These tests enumerate every combination of that invariant and verify
the coordinator correctly classifies each one.  No GPU, no LLM.
"""

import types
import uuid
import pytest

from verification.adversarial_search.schemas import (
    InputProposal, TensorDescriptor, ProposalVerdict, KernelExecutionResult,
)
from verification.adversarial_search.coordinator import SearchCoordinator


# ── Setup ─────────────────────────────────────────────────────────────────────

def _coord():
    """Minimal coordinator namespace for calling _evaluate_verdict."""
    c = types.SimpleNamespace()
    c._evaluate_verdict = SearchCoordinator._evaluate_verdict.__get__(c, SearchCoordinator)
    return c


def _proposal(pattern="partial_tile") -> InputProposal:
    return InputProposal(
        proposal_id=str(uuid.uuid4()),
        worker_id="w0",
        iteration=0,
        operator="softmax",
        tensors={"x": TensorDescriptor(shape=[4, 16], dtype="float32", fill="randn")},
        rationale="",
        predicted_failure_mode=pattern,
    )


def _result(proposal_id, kernel_id, passed_checker, passed_naive, check_results=None, error=None):
    return KernelExecutionResult(
        proposal_id=proposal_id,
        kernel_id=kernel_id,
        passed_checker=passed_checker,
        passed_naive=passed_naive,
        error=error,
        check_results=check_results or [],
        wall_time_ms=1.0,
    )


# ── Hit invariant ─────────────────────────────────────────────────────────────

class TestHitInvariant:
    """All three conditions must hold for is_hit=True."""

    def test_confirmed_hit(self):
        """ref passes + mutant fails checker + mutant passed naive → HIT."""
        c = _coord()
        p = _proposal()
        ref = _result(p.proposal_id, "reference", passed_checker=True,  passed_naive=True)
        mut = _result(p.proposal_id, "first_tile", passed_checker=False, passed_naive=True)
        v = c._evaluate_verdict(p, ref, [mut])
        assert v.is_hit
        assert v.gap_confirmed
        assert "first_tile" in v.hit_mutants

    def test_miss_reference_fails_checker(self):
        """ref fails checker → input is invalid → NOT a hit."""
        c = _coord()
        p = _proposal()
        ref = _result(p.proposal_id, "reference", passed_checker=False, passed_naive=False)
        mut = _result(p.proposal_id, "first_tile", passed_checker=False, passed_naive=True)
        v = c._evaluate_verdict(p, ref, [mut])
        assert not v.is_hit
        assert not v.reference_passed

    def test_miss_mutant_passes_checker(self):
        """mutant passes checker → bug not exposed → NOT a hit."""
        c = _coord()
        p = _proposal()
        ref = _result(p.proposal_id, "reference", passed_checker=True, passed_naive=True)
        mut = _result(p.proposal_id, "first_tile", passed_checker=True, passed_naive=True)
        v = c._evaluate_verdict(p, ref, [mut])
        assert not v.is_hit
        assert not v.gap_confirmed

    def test_miss_no_gap(self):
        """
        mutant fails checker BUT ALSO fails naive → gap not confirmed.
        Naive allclose would have caught it; our checker adds no value.
        NOT a hit.
        """
        c = _coord()
        p = _proposal()
        ref = _result(p.proposal_id, "reference", passed_checker=True,  passed_naive=True)
        mut = _result(p.proposal_id, "first_tile", passed_checker=False, passed_naive=False)
        v = c._evaluate_verdict(p, ref, [mut])
        assert not v.is_hit
        assert not v.gap_confirmed

    def test_partial_hit_some_mutants_caught(self):
        """
        Multiple mutants; some caught with gap, some not.
        Should be a HIT because at least one gap-confirmed mutant.
        """
        c = _coord()
        p = _proposal()
        ref  = _result(p.proposal_id, "reference",    True,  True)
        mut1 = _result(p.proposal_id, "first_tile",   False, True)   # gap!
        mut2 = _result(p.proposal_id, "wrong_reduction", False, False)  # no gap
        mut3 = _result(p.proposal_id, "boundary_mask",   True,  True)   # passes checker
        v = c._evaluate_verdict(p, ref, [mut1, mut2, mut3])
        assert v.is_hit
        assert "first_tile" in v.hit_mutants
        assert "wrong_reduction" in v.missed_mutants
        assert "boundary_mask" in v.missed_mutants

    def test_all_mutants_missed(self):
        c = _coord()
        p = _proposal()
        ref  = _result(p.proposal_id, "reference", True,  True)
        mut1 = _result(p.proposal_id, "m1",        True,  True)
        mut2 = _result(p.proposal_id, "m2",        False, False)
        v = c._evaluate_verdict(p, ref, [mut1, mut2])
        assert not v.is_hit
        assert set(v.missed_mutants) == {"m1", "m2"}
        assert v.hit_mutants == []

    def test_no_mutants(self):
        """Edge case: no mutant paths provided."""
        c = _coord()
        p = _proposal()
        ref = _result(p.proposal_id, "reference", True, True)
        v = c._evaluate_verdict(p, ref, [])
        assert not v.is_hit
        assert v.hit_mutants == []
        assert v.missed_mutants == []


# ── Verdict fields ────────────────────────────────────────────────────────────

class TestVerdictFields:
    def test_reference_passed_field(self):
        c = _coord()
        p = _proposal()
        ref_pass = _result(p.proposal_id, "reference", True,  True)
        ref_fail = _result(p.proposal_id, "reference", False, False)
        v_pass = c._evaluate_verdict(p, ref_pass, [])
        v_fail = c._evaluate_verdict(p, ref_fail, [])
        assert v_pass.reference_passed is True
        assert v_fail.reference_passed is False

    def test_hit_mutants_listed_correctly(self):
        c = _coord()
        p = _proposal()
        ref  = _result(p.proposal_id, "reference", True, True)
        muts = [
            _result(p.proposal_id, f"m{i}", passed_checker=(i % 2 == 0), passed_naive=True)
            for i in range(4)
        ]
        v = c._evaluate_verdict(p, ref, muts)
        # m1 and m3 fail checker and pass naive → should be hits
        assert "m1" in v.hit_mutants
        assert "m3" in v.hit_mutants
        # m0 and m2 pass checker → should be missed
        assert "m0" in v.missed_mutants
        assert "m2" in v.missed_mutants

    def test_failure_summary_not_empty(self):
        c = _coord()
        p = _proposal()
        ref = _result(p.proposal_id, "reference", True, True)
        mut = _result(p.proposal_id, "m1", False, False)
        v = c._evaluate_verdict(p, ref, [mut])
        assert len(v.failure_summary) > 0

    def test_proposal_id_propagated(self):
        c = _coord()
        p = _proposal()
        ref = _result(p.proposal_id, "reference", True, True)
        v = c._evaluate_verdict(p, ref, [])
        assert v.proposal_id == p.proposal_id


# ── Check-result integration ──────────────────────────────────────────────────

class TestCheckResultIntegration:
    def test_check_results_preserved_in_execution_result(self):
        """Verify check_results are stored on the result object."""
        check_results = [
            {"check_name": "nan_inf", "passed": True, "layer": "L1", "details": "ok"},
            {"check_name": "rows_sum_to_one", "passed": False, "layer": "L3",
             "details": "max_err=0.015"},
        ]
        r = KernelExecutionResult(
            proposal_id="p1",
            kernel_id="first_tile",
            passed_checker=False,
            passed_naive=True,
            error=None,
            check_results=check_results,
            wall_time_ms=2.0,
        )
        assert len(r.check_results) == 2
        assert r.check_results[1]["check_name"] == "rows_sum_to_one"