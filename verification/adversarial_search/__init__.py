"""
verification/adversarial_search/__init__.py

InputProposal/TensorDescriptor/ProposalVerdict/SearchResult stay eager
imports: schemas.py itself only depends on dataclasses/json/uuid, so
there's no cost to avoid there, and keeping them eager keeps this file
simpler than lazy-loading everything.
"""

from verification.adversarial_search.schemas import (
    InputProposal,
    TensorDescriptor,
    ProposalVerdict,
    SearchResult,
)

__all__ = [
    "SearchCoordinator",
    "InputProposal",
    "TensorDescriptor",
    "ProposalVerdict",
    "SearchResult",
    "get_strategy",
    "STRATEGIES",
]


def __getattr__(name):
    if name == "SearchCoordinator":
        from verification.adversarial_search.coordinator import SearchCoordinator
        return SearchCoordinator
    if name == "get_strategy":
        from verification.adversarial_search.strategy import get_strategy
        return get_strategy
    if name == "STRATEGIES":
        from verification.adversarial_search.strategy import STRATEGIES
        return STRATEGIES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")