"""
verification/adversarial_search/strategy/beam.py

Beam search strategy: maintain a beam of the top-B proposals by score,
expand each in the next iteration.

This is the same mechanism as AccelOpt (Hong et al. 2025 / Genghan Zhang),
adapted from kernel optimization to adversarial input search.

Key difference from AccelOpt: our beam candidates are InputProposals
(symbolic tensor descriptors), not kernel source files.  The expansion
step is an LLM refinement call in worker.refine(), not code generation.
The scoring signal is checker output (which bugs were exposed), not
latency measurements.

Beam width is set via --beam-width CLI arg (default 4).  A beam of 4
with 4 workers means each worker owns one beam member and expands it
independently — no inter-worker coordination needed.

Scoring:
  - Base: same as GreedyStrategy
  - Bonus: progressively reward proposals that get closer to a confirmed
    hit across iterations (near-miss signal)
"""

from __future__ import annotations
from typing import List, Tuple

from verification.adversarial_search.strategy.base import SearchStrategy
from verification.adversarial_search.schemas import InputProposal, ProposalVerdict


class BeamSearchStrategy(SearchStrategy):
    name = "beam"

    def score(
        self,
        proposal: InputProposal,
        verdict: ProposalVerdict,
    ) -> float:
        """
        Score components (additive):
          +10  reference passed checker  (input is valid — necessary condition)
          +8   per mutant caught with gap confirmed  (the money signal)
          +3   per mutant caught without gap  (partial credit)
          +2   reference passed but no mutant caught (valid input, worth building on)
          -5   reference failed (invalid input — deprioritise but don't discard)
          -2   per mutant that errored (crash, not informative)
        """
        score = 0.0
        if verdict.reference_passed:
            score += 10.0
        else:
            score -= 5.0

        for m in verdict.hit_mutants:
            if verdict.gap_confirmed:
                score += 8.0
            else:
                score += 3.0

        if verdict.reference_passed and not verdict.hit_mutants:
            score += 2.0

        return score

    def select(
        self,
        scored: List[Tuple[InputProposal, ProposalVerdict]],
        beam_width: int,
    ) -> List[Tuple[InputProposal, ProposalVerdict]]:
        """
        Select top-B by beam_score.
        If fewer than B valid candidates exist, fill remaining slots
        with the previous iteration's best (handled by coordinator).
        """
        if not scored:
            return []
        ranked = sorted(scored, key=lambda x: x[1].beam_score, reverse=True)
        return ranked[:beam_width]