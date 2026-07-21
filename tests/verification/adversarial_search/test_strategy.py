"""
tests/verification/adversarial_search/test_strategy.py

Tests for search strategies: greedy, beam, diverse beam.

Covers scoring, selection, beam width enforcement, diversity,
and the strategy registry.  No GPU, no LLM.
"""

import uuid
import pytest

from verification.adversarial_search.schemas import (
    InputProposal, TensorDescriptor, ProposalVerdict,
)
from verification.adversarial_search.strategy import (
    GreedyStrategy, BeamSearchStrategy, DiverseBeamStrategy,
    get_strategy, STRATEGIES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _verdict(
    proposal_id: str,
    reference_passed=True,
    hit_mutants=None,
    missed_mutants=None,
    gap_confirmed=False,
    beam_score=0.0,
) -> ProposalVerdict:
    hit_mutants = hit_mutants or []
    missed_mutants = missed_mutants or []
    is_hit = reference_passed and bool(hit_mutants) and gap_confirmed
    return ProposalVerdict(
        proposal_id=proposal_id,
        is_hit=is_hit,
        hit_mutants=hit_mutants,
        missed_mutants=missed_mutants,
        reference_passed=reference_passed,
        gap_confirmed=gap_confirmed,
        failure_summary="",
        beam_score=beam_score,
    )


def _make_pair(score=0.0, pattern="partial_tile", ref_pass=True, hit_mutants=None, gap=False):
    p = _proposal(pattern=pattern)
    v = _verdict(p.proposal_id, reference_passed=ref_pass,
                 hit_mutants=hit_mutants or [], gap_confirmed=gap, beam_score=score)
    return p, v


# ── Strategy registry ─────────────────────────────────────────────────────────

class TestStrategyRegistry:
    def test_all_strategies_registered(self):
        assert "greedy" in STRATEGIES
        assert "beam"   in STRATEGIES
        assert "diverse" in STRATEGIES

    def test_get_strategy_greedy(self):
        s = get_strategy("greedy")
        assert isinstance(s, GreedyStrategy)

    def test_get_strategy_beam(self):
        s = get_strategy("beam")
        assert isinstance(s, BeamSearchStrategy)

    def test_get_strategy_diverse(self):
        s = get_strategy("diverse", diversity_weight=5.0)
        assert isinstance(s, DiverseBeamStrategy)
        assert s.diversity_weight == 5.0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("mcts")


# ── GreedyStrategy ────────────────────────────────────────────────────────────

class TestGreedyStrategy:
    def setup_method(self):
        self.s = GreedyStrategy()

    def test_selects_single_best(self):
        pairs = [_make_pair(score=s) for s in [3.0, 15.0, 7.0, 1.0]]
        selected = self.s.select(pairs, beam_width=4)
        assert len(selected) == 1
        assert selected[0][1].beam_score == 15.0

    def test_empty_input(self):
        assert self.s.select([], beam_width=4) == []

    def test_single_element(self):
        pairs = [_make_pair(score=5.0)]
        selected = self.s.select(pairs, beam_width=4)
        assert len(selected) == 1

    def test_score_reference_passes(self):
        p, v = _make_pair(ref_pass=True)
        score = self.s.score(p, v)
        assert score > 0

    def test_score_reference_fails_lower(self):
        p_pass, v_pass = _make_pair(ref_pass=True)
        p_fail, v_fail = _make_pair(ref_pass=False)
        assert self.s.score(p_pass, v_pass) >= self.s.score(p_fail, v_fail)

    def test_score_hit_higher_than_miss(self):
        p_hit,  v_hit  = _make_pair(ref_pass=True, hit_mutants=["m1"], gap=True)
        p_miss, v_miss = _make_pair(ref_pass=True)
        assert self.s.score(p_hit, v_hit) > self.s.score(p_miss, v_miss)

    def test_ignores_beam_width(self):
        """Greedy always returns 1 regardless of beam_width."""
        pairs = [_make_pair(score=float(i)) for i in range(10)]
        for bw in [1, 2, 4, 8]:
            selected = self.s.select(pairs, beam_width=bw)
            assert len(selected) == 1


# ── BeamSearchStrategy ────────────────────────────────────────────────────────

class TestBeamSearchStrategy:
    def setup_method(self):
        self.s = BeamSearchStrategy()

    def test_selects_top_b(self):
        scores = [1.0, 9.0, 7.0, 3.0, 5.0]
        pairs = [_make_pair(score=s) for s in scores]
        selected = self.s.select(pairs, beam_width=3)
        assert len(selected) == 3
        selected_scores = sorted([x[1].beam_score for x in selected], reverse=True)
        assert selected_scores == [9.0, 7.0, 5.0]

    def test_beam_width_1(self):
        pairs = [_make_pair(score=float(i)) for i in range(5)]
        selected = self.s.select(pairs, beam_width=1)
        assert len(selected) == 1
        assert selected[0][1].beam_score == 4.0

    def test_fewer_candidates_than_beam_width(self):
        pairs = [_make_pair(score=float(i)) for i in range(2)]
        selected = self.s.select(pairs, beam_width=10)
        assert len(selected) == 2

    def test_empty_input(self):
        assert self.s.select([], beam_width=4) == []

    def test_score_hit_with_gap_highest(self):
        p, v = _make_pair(ref_pass=True, hit_mutants=["m1"], gap=True)
        score = self.s.score(p, v)
        # Hit with gap: +10 (ref) + 8 (hit+gap) = 18
        assert score >= 18.0

    def test_score_ref_fails_penalised(self):
        p, v = _make_pair(ref_pass=False)
        score = self.s.score(p, v)
        assert score < 0

    def test_score_valid_input_no_mutant_caught(self):
        p, v = _make_pair(ref_pass=True, hit_mutants=[])
        score = self.s.score(p, v)
        # ref passes (+10) + no mutants caught (+2 bonus) = 12
        assert score > 0

    def test_selected_sorted_descending(self):
        scores = [2.0, 8.0, 5.0, 1.0, 9.0, 3.0]
        pairs = [_make_pair(score=s) for s in scores]
        selected = self.s.select(pairs, beam_width=4)
        selected_scores = [x[1].beam_score for x in selected]
        assert selected_scores == sorted(selected_scores, reverse=True)


# ── DiverseBeamStrategy ───────────────────────────────────────────────────────

class TestDiverseBeamStrategy:
    def setup_method(self):
        self.s = DiverseBeamStrategy(diversity_weight=10.0)

    def test_promotes_diverse_patterns(self):
        """High-scoring duplicate pattern should not fill all beam slots."""
        patterns_scores = [
            ("partial_tile",  10.0),
            ("partial_tile",   9.0),
            ("wrong_reduction", 7.0),
            ("boundary_mask",   6.0),
        ]
        pairs = [_make_pair(score=s, pattern=p) for p, s in patterns_scores]
        selected = self.s.select(pairs, beam_width=3)
        sel_patterns = [x[0].predicted_failure_mode for x in selected]
        # With high diversity weight, second partial_tile should be displaced
        assert "wrong_reduction" in sel_patterns or "boundary_mask" in sel_patterns

    def test_always_keeps_best(self):
        """Best-scoring candidate always included regardless of pattern duplication."""
        pairs = [_make_pair(score=100.0, pattern="partial_tile"),
                 _make_pair(score=1.0,   pattern="wrong_reduction")]
        selected = self.s.select(pairs, beam_width=2)
        best_scores = [x[1].beam_score for x in selected]
        assert 100.0 in best_scores

    def test_fills_beam_when_all_same_pattern(self):
        """Beam must be filled even if all candidates share a pattern."""
        pairs = [_make_pair(score=float(i), pattern="partial_tile") for i in range(6)]
        selected = self.s.select(pairs, beam_width=4)
        assert len(selected) == 4

    def test_empty_input(self):
        assert self.s.select([], beam_width=4) == []

    def test_diversity_weight_zero_same_as_beam(self):
        """λ=0 should behave identically to plain beam search."""
        s_diverse = DiverseBeamStrategy(diversity_weight=0.0)
        s_beam    = BeamSearchStrategy()
        scores = [5.0, 9.0, 7.0, 3.0]
        pairs = [_make_pair(score=s, pattern="partial_tile") for s in scores]
        sel_d = s_diverse.select(pairs, beam_width=2)
        sel_b = s_beam.select(pairs, beam_width=2)
        assert [x[1].beam_score for x in sel_d] == [x[1].beam_score for x in sel_b]

    def test_default_diversity_weight(self):
        s = DiverseBeamStrategy()
        assert s.diversity_weight == 3.0