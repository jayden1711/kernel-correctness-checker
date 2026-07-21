"""
tests/verification/adversarial_search/test_schemas.py

Tests for schemas.py: typed contracts, validation, and round-trip fidelity.

These tests have zero external dependencies (no GPU, no LLM, no Triton).
They are the fastest feedback loop for schema changes.
"""

import json
import uuid
import pytest
import torch

from verification.adversarial_search.schemas import (
    TensorDescriptor,
    InputProposal,
    ExecutionError,
    KernelExecutionResult,
    ProposalVerdict,
    WorkerFeedback,
    SearchResult,
    validate_proposal,
    REQUIRED_TENSOR_KEYS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_descriptor(**kwargs):
    defaults = dict(shape=[4, 16], dtype="float32", fill="randn", scale=1.0, shift=0.0)
    defaults.update(kwargs)
    return TensorDescriptor(**defaults)


def _make_proposal(operator="softmax", extra_tensors=None, **kwargs):
    tensors = {"x": _make_descriptor()}
    if extra_tensors:
        tensors.update(extra_tensors)
    return InputProposal(
        proposal_id=str(uuid.uuid4()),
        worker_id="test-worker",
        iteration=0,
        operator=operator,
        tensors=tensors,
        rationale="test rationale",
        predicted_failure_mode="partial_tile",
        **kwargs,
    )


# ── TensorDescriptor ──────────────────────────────────────────────────────────

class TestTensorDescriptor:
    def test_roundtrip_basic(self):
        d = _make_descriptor()
        assert TensorDescriptor.from_dict(d.to_dict()) == d

    def test_roundtrip_with_patches(self):
        d = _make_descriptor(
            patches=[{"indices": "[:, -1]", "value": 1e9},
                     {"indices": "[0, :]", "value": -1.0}]
        )
        recovered = TensorDescriptor.from_dict(d.to_dict())
        assert recovered.patches == d.patches

    def test_roundtrip_literal(self):
        d = _make_descriptor(fill="literal", literal_values=[1.0, 2.0, 3.0, 4.0], shape=[2, 2])
        recovered = TensorDescriptor.from_dict(d.to_dict())
        assert recovered.literal_values == [1.0, 2.0, 3.0, 4.0]

    def test_to_dict_is_json_serialisable(self):
        d = _make_descriptor(patches=[{"indices": "[:, -8:]", "value": 1e4}])
        s = json.dumps(d.to_dict())
        assert isinstance(s, str)

    def test_defaults(self):
        d = TensorDescriptor(shape=[2, 2], dtype="float32", fill="ones")
        assert d.scale == 1.0
        assert d.shift == 0.0
        assert d.patches == []
        assert d.literal_values is None


# ── InputProposal ─────────────────────────────────────────────────────────────

class TestInputProposal:
    def test_roundtrip(self):
        p = _make_proposal()
        recovered = InputProposal.from_dict(p.to_dict())
        assert recovered.proposal_id == p.proposal_id
        assert recovered.operator == p.operator
        assert recovered.tensors["x"].shape == p.tensors["x"].shape

    def test_roundtrip_preserves_patches(self):
        p = _make_proposal()
        p.tensors["x"].patches = [{"indices": "[:, -32:]", "value": 1e4}]
        recovered = InputProposal.from_dict(p.to_dict())
        assert recovered.tensors["x"].patches == [{"indices": "[:, -32:]", "value": 1e4}]

    def test_to_json_and_back(self):
        p = _make_proposal()
        recovered = InputProposal.from_dict(json.loads(p.to_json()))
        assert recovered.proposal_id == p.proposal_id

    def test_score_default(self):
        p = _make_proposal()
        assert p.score == 0.0

    def test_roundtrip_preserves_score(self):
        p = _make_proposal(score=7.5)
        recovered = InputProposal.from_dict(p.to_dict())
        assert recovered.score == 7.5

    @pytest.mark.parametrize("operator", list(REQUIRED_TENSOR_KEYS.keys()))
    def test_operator_roundtrip(self, operator):
        tensors = {
            k: _make_descriptor()
            for k in REQUIRED_TENSOR_KEYS[operator]
        }
        p = InputProposal(
            proposal_id=str(uuid.uuid4()),
            worker_id="w0",
            iteration=0,
            operator=operator,
            tensors=tensors,
            rationale="r",
            predicted_failure_mode="p",
        )
        recovered = InputProposal.from_dict(p.to_dict())
        assert set(recovered.tensors.keys()) == set(tensors.keys())


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidateProposal:
    def test_valid_softmax(self):
        p = _make_proposal(operator="softmax")
        ok, msg = validate_proposal(p)
        assert ok, msg

    def test_valid_layernorm(self):
        p = _make_proposal(
            operator="layernorm",
            extra_tensors={
                "gamma": _make_descriptor(shape=[16]),
                "beta":  _make_descriptor(shape=[16]),
            }
        )
        ok, msg = validate_proposal(p)
        assert ok, msg

    def test_valid_rmsnorm(self):
        p = _make_proposal(
            operator="rmsnorm",
            extra_tensors={"gamma": _make_descriptor(shape=[16])}
        )
        ok, msg = validate_proposal(p)
        assert ok, msg

    def test_missing_key_layernorm(self):
        p = _make_proposal(operator="layernorm")
        # Missing gamma and beta
        ok, msg = validate_proposal(p)
        assert not ok
        assert "gamma" in msg or "beta" in msg

    def test_missing_key_matmul(self):
        p = _make_proposal(operator="matmul")
        # Missing B
        ok, msg = validate_proposal(p)
        assert not ok
        assert "B" in msg

    def test_unknown_operator(self):
        p = _make_proposal(operator="unknown_op")
        ok, msg = validate_proposal(p)
        assert not ok
        assert "Unknown operator" in msg

    def test_empty_shape_rejected(self):
        p = _make_proposal()
        p.tensors["x"].shape = []
        ok, msg = validate_proposal(p)
        assert not ok
        assert "shape" in msg.lower() or "empty" in msg.lower()

    @pytest.mark.parametrize("bad_fill", ["gaussian", "uniform", "xavier", "normal", ""])
    def test_invalid_fill_rejected(self, bad_fill):
        p = _make_proposal()
        p.tensors["x"].fill = bad_fill
        ok, msg = validate_proposal(p)
        assert not ok
        assert "fill" in msg.lower()

    def test_literal_without_values_rejected(self):
        p = _make_proposal()
        p.tensors["x"].fill = "literal"
        p.tensors["x"].literal_values = None
        ok, msg = validate_proposal(p)
        assert not ok
        assert "literal" in msg.lower()

    @pytest.mark.parametrize("valid_fill", ["randn", "ones", "zeros", "arange"])
    def test_valid_fills_accepted(self, valid_fill):
        p = _make_proposal()
        p.tensors["x"].fill = valid_fill
        ok, _ = validate_proposal(p)
        assert ok


# ── KernelExecutionResult ─────────────────────────────────────────────────────

class TestKernelExecutionResult:
    def test_roundtrip_no_error(self):
        r = KernelExecutionResult(
            proposal_id=str(uuid.uuid4()),
            kernel_id="reference",
            passed_checker=True,
            passed_naive=True,
            error=None,
            check_results=[{"check_name": "nan_inf", "passed": True, "layer": "L1", "details": "ok"}],
            wall_time_ms=12.3,
        )
        recovered = KernelExecutionResult.from_dict(r.to_dict())
        assert recovered.passed_checker == r.passed_checker
        assert recovered.wall_time_ms == r.wall_time_ms
        assert recovered.error is None

    def test_roundtrip_with_error(self):
        err = ExecutionError(
            error_type="ValueError",
            message="max_err exceeded",
            layer="L2",
            check_name="perturbation_tolerance",
            max_err=0.12,
            traceback_snippet="...",
        )
        r = KernelExecutionResult(
            proposal_id=str(uuid.uuid4()),
            kernel_id="first_tile",
            passed_checker=False,
            passed_naive=False,
            error=err,
            check_results=[],
            wall_time_ms=0.0,
        )
        recovered = KernelExecutionResult.from_dict(r.to_dict())
        assert recovered.error is not None
        assert recovered.error.error_type == "ValueError"
        assert recovered.error.max_err == 0.12


# ── ProposalVerdict ───────────────────────────────────────────────────────────

class TestProposalVerdict:
    def test_roundtrip(self):
        v = ProposalVerdict(
            proposal_id=str(uuid.uuid4()),
            is_hit=True,
            hit_mutants=["first_tile"],
            missed_mutants=["wrong_reduction"],
            reference_passed=True,
            gap_confirmed=True,
            failure_summary="caught first_tile",
            beam_score=12.5,
        )
        recovered = ProposalVerdict.from_dict(v.to_dict())
        assert recovered.is_hit == v.is_hit
        assert recovered.hit_mutants == v.hit_mutants
        assert recovered.beam_score == v.beam_score

    def test_hit_requires_gap(self):
        """Ensure gap_confirmed is not set when it shouldn't be."""
        v = ProposalVerdict(
            proposal_id="p1",
            is_hit=False,
            hit_mutants=[],
            missed_mutants=["m1"],
            reference_passed=True,
            gap_confirmed=False,
            failure_summary="no hit",
        )
        assert not v.is_hit
        assert not v.gap_confirmed