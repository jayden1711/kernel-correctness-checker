"""
tests/verification/adversarial_search/test_prompts.py

Tests for prompts/base.py: prompt content, formatting, and operator coverage.

No GPU, no LLM.
"""

import uuid
import pytest

from verification.adversarial_search.schemas import (
    InputProposal, TensorDescriptor, ProposalVerdict, WorkerFeedback,
)
from verification.adversarial_search.prompts.base import (
    SYSTEM_PROMPT,
    OPERATOR_CONTEXT,
    BUG_PATTERN_HINTS,
    format_first_turn,
    format_refine_turn,
)


# ── System prompt ─────────────────────────────────────────────────────────────

class TestSystemPrompt:
    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_json_schema_mentioned(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_required_fields_in_schema(self):
        for field in ["rationale", "predicted_failure_mode", "tensors"]:
            assert field in SYSTEM_PROMPT

    def test_fill_strategies_documented(self):
        for fill in ["randn", "ones", "zeros", "arange"]:
            assert fill in SYSTEM_PROMPT

    def test_patches_documented(self):
        assert "patches" in SYSTEM_PROMPT


# ── Operator context coverage ─────────────────────────────────────────────────

class TestOperatorContext:
    def test_all_operators_covered(self):
        required = {"softmax", "layernorm", "matmul", "flash_attention", "rmsnorm"}
        assert required.issubset(set(OPERATOR_CONTEXT.keys()))

    @pytest.mark.parametrize("operator", ["softmax", "layernorm", "matmul", "flash_attention", "rmsnorm"])
    def test_operator_context_not_empty(self, operator):
        ctx = OPERATOR_CONTEXT[operator]
        assert len(ctx) > 50

    @pytest.mark.parametrize("operator", ["softmax", "layernorm", "matmul", "flash_attention", "rmsnorm"])
    def test_operator_context_mentions_required_tensors(self, operator):
        ctx = OPERATOR_CONTEXT[operator]
        tensor_names = {
            "softmax":         ["x"],
            "layernorm":       ["x", "gamma", "beta"],
            "matmul":          ["A", "B"],
            "flash_attention": ["Q", "K", "V"],
            "rmsnorm":         ["x", "gamma"],
        }[operator]
        for t in tensor_names:
            assert t in ctx, f"'{t}' not in {operator} context"

    @pytest.mark.parametrize("operator", ["softmax", "layernorm", "matmul", "flash_attention", "rmsnorm"])
    def test_operator_context_mentions_common_bugs(self, operator):
        ctx = OPERATOR_CONTEXT[operator]
        # Each operator context should mention at least one bug pattern
        assert "bug" in ctx.lower() or "common" in ctx.lower() or "error" in ctx.lower()


# ── Bug pattern hints ─────────────────────────────────────────────────────────

class TestBugPatternHints:
    def test_all_known_patterns_have_hints(self):
        expected_patterns = {
            "partial_tile", "wrong_reduction", "dtype_overflow",
            "boundary_mask", "wrong_axis", "numerical_instab",
            "ignore_gamma_beta", "ignore_gamma", "partial_reduction",
            "skip_mean_subtract", "wrong_variance", "drop_last_tile",
            "skip_rescaling", "approx_denom", "wrong_mask", "missing_eps",
            "wrong_norm", "partial_k_reduct", "skip_boundary",
            "swapped_strides", "wrong_dtype",
        }
        for pattern in expected_patterns:
            assert pattern in BUG_PATTERN_HINTS, f"Missing hint for {pattern!r}"

    @pytest.mark.parametrize("pattern", [
        "partial_tile", "wrong_reduction", "ignore_gamma",
        "drop_last_tile", "partial_k_reduct",
    ])
    def test_hints_are_actionable(self, pattern):
        """Hints should contain either a shape, a value, or a patch."""
        hint = BUG_PATTERN_HINTS[pattern]
        has_content = (
            "shape" in hint or "patches" in hint or "fill" in hint
            or "scale" in hint or "value" in hint or any(c.isdigit() for c in hint)
        )
        assert has_content, f"Hint for {pattern!r} seems too vague: {hint!r}"


# ── format_first_turn ─────────────────────────────────────────────────────────

class TestFormatFirstTurn:
    @pytest.mark.parametrize("operator", ["softmax", "layernorm", "rmsnorm", "matmul", "flash_attention"])
    def test_includes_operator_name(self, operator):
        msg = format_first_turn(operator)
        assert operator in msg

    def test_includes_operator_context(self):
        msg = format_first_turn("softmax")
        # The operator context should be embedded
        assert "n_rows" in msg or "softmax" in msg.lower()

    def test_seed_pattern_included(self):
        msg = format_first_turn("softmax", seed_bug_pattern="partial_tile")
        assert "partial_tile" in msg

    def test_seed_pattern_hint_included(self):
        msg = format_first_turn("softmax", seed_bug_pattern="partial_tile")
        # BUG_PATTERN_HINTS["partial_tile"] contains "-32:" or similar
        assert "tile" in msg.lower() or "spike" in msg.lower() or "-32" in msg or "patch" in msg.lower()

    def test_no_seed_pattern(self):
        msg = format_first_turn("softmax", seed_bug_pattern=None)
        assert "softmax" in msg
        assert len(msg) > 50

    def test_json_schema_reminder(self):
        msg = format_first_turn("softmax")
        assert "JSON" in msg


# ── format_refine_turn ────────────────────────────────────────────────────────

class TestFormatRefineTurn:
    def _make_feedback(self, is_hit=False, ref_pass=True, hints=None, memory=None):
        pid = str(uuid.uuid4())
        v = ProposalVerdict(
            proposal_id=pid,
            is_hit=is_hit,
            hit_mutants=["m1"] if is_hit else [],
            missed_mutants=[] if is_hit else ["m1"],
            reference_passed=ref_pass,
            gap_confirmed=is_hit,
            failure_summary="test summary",
        )
        return WorkerFeedback(
            proposal_id=pid,
            verdict=v,
            hints=hints or ["Try larger values in the last tile."],
            memory_items=memory or [],
        )

    def test_includes_status(self):
        fb = self._make_feedback(is_hit=False, ref_pass=True)
        msg = format_refine_turn(fb)
        assert "MISS" in msg or "miss" in msg.lower()

    def test_includes_hit_status(self):
        fb = self._make_feedback(is_hit=True, ref_pass=True)
        msg = format_refine_turn(fb)
        assert "HIT" in msg

    def test_reference_failed_status(self):
        fb = self._make_feedback(is_hit=False, ref_pass=False)
        msg = format_refine_turn(fb)
        assert "reference" in msg.lower()

    def test_hints_included(self):
        fb = self._make_feedback(hints=["Try spike in last tile.", "Reduce magnitude by 10x."])
        msg = format_refine_turn(fb)
        assert "spike" in msg.lower() or "last tile" in msg.lower()
        assert "magnitude" in msg.lower() or "10x" in msg.lower()

    def test_memory_items_included(self):
        fb = self._make_feedback(memory=["[partial_tile] Spike at [:,-32:] worked."])
        msg = format_refine_turn(fb)
        assert "partial_tile" in msg or "spike" in msg.lower() or "memory" in msg.lower()

    def test_no_memory_items_no_crash(self):
        fb = self._make_feedback(memory=[])
        msg = format_refine_turn(fb)
        assert len(msg) > 50

    def test_json_reminder_present(self):
        fb = self._make_feedback()
        msg = format_refine_turn(fb)
        assert "JSON" in msg

    def test_proposal_id_included(self):
        fb = self._make_feedback()
        msg = format_refine_turn(fb)
        assert fb.proposal_id[:8] in msg