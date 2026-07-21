"""
verification/adversarial_search/strategy/greedy.py

Greedy strategy: keep the single highest-scoring proposal from each
iteration and use it as the seed for the next.

Advantages: lowest LLM cost, fastest wall time per iteration.
Disadvantages: premature convergence, misses bugs that require diverse
input structures.

Use for: quick sanity checks, single-worker runs, very tight budgets.
"""

from __future__ import annotations
from typing import List, Tuple

from verification.adversarial_search.strategy.base import SearchStrategy
from verification.adversarial_search.schemas import InputProposal, ProposalVerdict


class GreedyStrategy(SearchStrategy):
    name = "greedy"

    def score(
        self,
        proposal: InputProposal,
        verdict: ProposalVerdict,
    ) -> float:
        """
        Score components (additive):
          +10  reference passed checker  (input is valid)
          +5   per mutant that failed checker but passed naive (gap confirmed)
          +2   per mutant that failed checker (even without gap)
          -3   per mutant that errored (crashed, not useful signal)
          +1   gap_confirmed bonus
        """
        score = 0.0
        if verdict.reference_passed:
            score += 10.0
        score += 5.0 * len([m for m in verdict.hit_mutants])
        if verdict.gap_confirmed:
            score += 1.0
        return score

    def select(
        self,
        scored: List[Tuple[InputProposal, ProposalVerdict]],
        beam_width: int,
    ) -> List[Tuple[InputProposal, ProposalVerdict]]:
        if not scored:
            return []
        return [max(scored, key=lambda x: x[1].beam_score)]