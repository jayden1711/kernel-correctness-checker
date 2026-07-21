"""
verification/adversarial_search/strategy/diverse.py

Diverse beam search: beam search with an explicit diversity penalty
that discourages multiple beam members from targeting the same bug
pattern.

Motivation: plain beam search tends to collapse onto the highest-
scoring region of the search space.  For adversarial input search,
this means all workers pile onto the same bug pattern (e.g. partial_tile)
while other patterns (e.g. dtype_overflow) go unexplored.

Mechanism: after scoring, apply a penalty to any candidate whose
predicted_failure_mode already appears in the current beam.  This
encourages the beam to cover multiple distinct failure modes, which:
  1. Maximises the number of distinct mutants caught per search run
  2. Produces a more diverse result set for the paper's coverage table
  3. Reduces the risk of all workers converging on an input that exposes
     the same mutant via different paths

This is the KernelAgent "beam_search_diverse" config pattern adapted
to our search space.

Diversity weight λ is configurable (default 3.0).  Higher λ = more
diverse but slower convergence on any single pattern.
"""

from __future__ import annotations
from typing import List, Tuple

from verification.adversarial_search.strategy.beam import BeamSearchStrategy
from verification.adversarial_search.schemas import InputProposal, ProposalVerdict


class DiverseBeamStrategy(BeamSearchStrategy):
    name = "diverse"

    def __init__(self, diversity_weight: float = 3.0):
        self.diversity_weight = diversity_weight

    def select(
        self,
        scored: List[Tuple[InputProposal, ProposalVerdict]],
        beam_width: int,
    ) -> List[Tuple[InputProposal, ProposalVerdict]]:
        """
        Greedy diverse selection:
          1. Pick the highest-scoring candidate unconditionally (best first)
          2. For each subsequent slot, penalise candidates whose
             predicted_failure_mode already appears in the selected set
          3. Repeat until beam is full
        """
        if not scored:
            return []

        ranked = sorted(scored, key=lambda x: x[1].beam_score, reverse=True)
        selected: List[Tuple[InputProposal, ProposalVerdict]] = []
        selected_patterns: List[str] = []

        for proposal, verdict in ranked:
            if len(selected) >= beam_width:
                break
            pattern = proposal.predicted_failure_mode
            # Count how many times this pattern already appears in the beam
            overlap = selected_patterns.count(pattern)
            # Apply penalty to the effective score for selection purposes
            effective_score = verdict.beam_score - overlap * self.diversity_weight
            # Always accept if beam is empty or effective_score is positive enough
            if not selected or effective_score > 0 or len(selected) < beam_width // 2:
                selected.append((proposal, verdict))
                selected_patterns.append(pattern)

        # If we didn't fill the beam (all remaining had high overlap), fill with top-scored
        remaining = [x for x in ranked if x not in selected]
        while len(selected) < beam_width and remaining:
            selected.append(remaining.pop(0))

        return selected