"""
tests/verification/adversarial_search/test_worker_parsing.py

Tests for worker.py: JSON parsing, schema validation, retry on failure,
and history trimming.

LLM calls are mocked — no API key required.
"""

import json
import uuid
import pytest
from unittest.mock import patch, MagicMock

from verification.adversarial_search.worker import AdversarialWorker
from verification.adversarial_search.schemas import InputProposal, validate_proposal


# ── Valid LLM response templates ──────────────────────────────────────────────

VALID_SOFTMAX_RESPONSE = json.dumps({
    "rationale": "Spike in last tile exposes first-tile-only processing bug.",
    "predicted_failure_mode": "partial_tile",
    "tensors": {
        "x": {
            "shape": [128, 256],
            "dtype": "float32",
            "fill": "randn",
            "scale": 1.0,
            "shift": 0.0,
            "patches": [{"indices": "[:, -32:]", "value": 1e4}],
        }
    }
})

VALID_LAYERNORM_RESPONSE = json.dumps({
    "rationale": "Non-unit gamma exposes ignore_gamma bug.",
    "predicted_failure_mode": "ignore_gamma_beta",
    "tensors": {
        "x":     {"shape": [64, 128], "dtype": "float32", "fill": "randn",
                  "scale": 1.0, "shift": 0.0, "patches": []},
        "gamma": {"shape": [128], "dtype": "float32", "fill": "ones",
                  "scale": 2.0, "shift": 0.0, "patches": []},
        "beta":  {"shape": [128], "dtype": "float32", "fill": "zeros",
                  "scale": 1.0, "shift": 3.0, "patches": []},
    }
})


def _mock_llm(response_text: str):
    """Return a mock litellm completion response."""
    msg = MagicMock()
    msg.content = response_text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _worker(operator="softmax") -> AdversarialWorker:
    return AdversarialWorker(
        worker_id="w-test",
        operator=operator,
        model="claude-sonnet-4-6",
        seed_bug_pattern="partial_tile",
    )


# ── Parsing ───────────────────────────────────────────────────────────────────

class TestWorkerParsing:
    @patch("litellm.completion")
    def test_parse_valid_softmax_response(self, mock_completion):
        mock_completion.return_value = _mock_llm(VALID_SOFTMAX_RESPONSE)
        w = _worker("softmax")
        proposal = w.propose()
        assert isinstance(proposal, InputProposal)
        assert proposal.operator == "softmax"
        assert "x" in proposal.tensors
        assert proposal.tensors["x"].shape == [128, 256]
        assert proposal.tensors["x"].patches == [{"indices": "[:, -32:]", "value": 1e4}]

    @patch("litellm.completion")
    def test_parse_valid_layernorm_response(self, mock_completion):
        mock_completion.return_value = _mock_llm(VALID_LAYERNORM_RESPONSE)
        w = _worker("layernorm")
        proposal = w.propose()
        ok, msg = validate_proposal(proposal)
        assert ok, msg
        assert proposal.tensors["gamma"].scale == 2.0

    @patch("litellm.completion")
    def test_strips_markdown_fences(self, mock_completion):
        fenced = f"```json\n{VALID_SOFTMAX_RESPONSE}\n```"
        mock_completion.return_value = _mock_llm(fenced)
        w = _worker("softmax")
        proposal = w.propose()
        assert proposal.operator == "softmax"

    @patch("litellm.completion")
    def test_strips_triple_backtick_no_language(self, mock_completion):
        fenced = f"```\n{VALID_SOFTMAX_RESPONSE}\n```"
        mock_completion.return_value = _mock_llm(fenced)
        w = _worker("softmax")
        proposal = w.propose()
        assert proposal.operator == "softmax"

    @patch("litellm.completion")
    def test_proposal_id_is_uuid(self, mock_completion):
        mock_completion.return_value = _mock_llm(VALID_SOFTMAX_RESPONSE)
        w = _worker("softmax")
        proposal = w.propose()
        # Should not raise
        uuid.UUID(proposal.proposal_id)

    @patch("litellm.completion")
    def test_worker_id_propagated(self, mock_completion):
        mock_completion.return_value = _mock_llm(VALID_SOFTMAX_RESPONSE)
        w = _worker("softmax")
        proposal = w.propose()
        assert proposal.worker_id == "w-test"

    @patch("litellm.completion")
    def test_iteration_counter_increments(self, mock_completion):
        mock_completion.return_value = _mock_llm(VALID_SOFTMAX_RESPONSE)
        w = _worker("softmax")
        p1 = w.propose()
        assert p1.iteration == 0
        # Mock feedback for refinement
        from verification.adversarial_search.schemas import (
            ProposalVerdict, WorkerFeedback,
        )
        v = ProposalVerdict(p1.proposal_id, False, [], [], True, False, "")
        fb = WorkerFeedback(p1.proposal_id, v, ["try again"])
        mock_completion.return_value = _mock_llm(VALID_SOFTMAX_RESPONSE)
        p2 = w.refine(fb)
        assert p2.iteration == 1


# ── Error handling and retry ──────────────────────────────────────────────────

class TestWorkerRetry:
    @patch("litellm.completion")
    def test_invalid_json_triggers_retry(self, mock_completion):
        """First call returns invalid JSON, second returns valid."""
        valid = _mock_llm(VALID_SOFTMAX_RESPONSE)
        invalid = _mock_llm("not valid json at all {{{")
        mock_completion.side_effect = [invalid, valid]
        w = _worker("softmax")
        proposal = w.propose()
        assert proposal.operator == "softmax"
        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    def test_missing_required_field_triggers_retry(self, mock_completion):
        """Response missing 'tensors' key should trigger retry."""
        bad_response = json.dumps({
            "rationale": "test",
            "predicted_failure_mode": "partial_tile",
            # missing "tensors"
        })
        valid = _mock_llm(VALID_SOFTMAX_RESPONSE)
        mock_completion.side_effect = [_mock_llm(bad_response), valid]
        w = _worker("softmax")
        proposal = w.propose()
        assert "x" in proposal.tensors

    @patch("litellm.completion")
    def test_all_retries_exhausted_raises(self, mock_completion):
        """If all MAX_RETRIES+1 calls fail, RuntimeError is raised."""
        bad = _mock_llm("not json {")
        mock_completion.return_value = bad
        w = _worker("softmax")
        with pytest.raises(RuntimeError, match="failed to produce valid JSON"):
            w.propose()

    @patch("litellm.completion")
    def test_wrong_operator_tensor_keys_triggers_retry(self, mock_completion):
        """Response with wrong tensor keys for operator should retry."""
        wrong_keys = json.dumps({
            "rationale": "test",
            "predicted_failure_mode": "partial_tile",
            "tensors": {
                "Q": {"shape": [32, 64], "dtype": "float32", "fill": "randn",
                      "scale": 1.0, "shift": 0.0, "patches": []},
            }
        })
        valid = _mock_llm(VALID_SOFTMAX_RESPONSE)
        mock_completion.side_effect = [_mock_llm(wrong_keys), valid]
        w = _worker("softmax")
        proposal = w.propose()
        assert "x" in proposal.tensors


# ── History trimming ──────────────────────────────────────────────────────────

class TestWorkerHistoryTrimming:
    @patch("litellm.completion")
    def test_history_does_not_grow_unboundedly(self, mock_completion):
        """History must stay at most MAX_HISTORY_TURNS messages."""
        mock_completion.return_value = _mock_llm(VALID_SOFTMAX_RESPONSE)
        w = _worker("softmax")
        from verification.adversarial_search.schemas import (
            ProposalVerdict, WorkerFeedback,
        )
        # Propose + refine many times
        p = w.propose()
        for i in range(20):
            v = ProposalVerdict(p.proposal_id, False, [], [], True, False, "")
            fb = WorkerFeedback(p.proposal_id, v, ["hint"])
            p = w.refine(fb)
        assert len(w._history) <= w.MAX_HISTORY_TURNS