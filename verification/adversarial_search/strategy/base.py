"""
verification/adversarial_search/strategy/base.py

Abstract base class for search strategies.

A strategy controls two things:
  1. select(verdicts, beam_width) → which proposals to carry forward
  2. score(proposal, verdict)     → numeric score for ranking

This keeps the coordinator strategy-agnostic.  Swap strategies via
--strategy flag without touching coordinator or worker code.

Available strategies:
  greedy      — keep the single best proposal per iteration, expand it
  beam        — keep top-B proposals, expand each in parallel (AccelOpt pattern)
  diverse     — beam search with explicit diversity penalty (KernelAgent pattern)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple

from verification.adversarial_search.schemas import InputProposal, ProposalVerdict


class SearchStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def score(
        self,
        proposal: InputProposal,
        verdict: ProposalVerdict,
    ) -> float:
        """
        Assign a numeric score to a (proposal, verdict) pair.
        Higher = more promising to expand in the next iteration.
        Called by the coordinator after every verdict.
        """

    @abstractmethod
    def select(
        self,
        scored: List[Tuple[InputProposal, ProposalVerdict]],
        beam_width: int,
    ) -> List[Tuple[InputProposal, ProposalVerdict]]:
        """
        Given all (proposal, verdict) pairs from this iteration,
        return the subset to carry forward as beam members.
        beam_width is always passed even if the strategy ignores it.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"