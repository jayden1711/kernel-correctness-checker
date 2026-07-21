from verification.adversarial_search.strategy.base import SearchStrategy
from verification.adversarial_search.strategy.greedy import GreedyStrategy
from verification.adversarial_search.strategy.beam import BeamSearchStrategy
from verification.adversarial_search.strategy.diverse import DiverseBeamStrategy

STRATEGIES = {
    "greedy":  GreedyStrategy,
    "beam":    BeamSearchStrategy,
    "diverse": DiverseBeamStrategy,
}

def get_strategy(name: str, **kwargs) -> SearchStrategy:
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy: {name!r}. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    return cls(**kwargs)

__all__ = [
    "SearchStrategy",
    "GreedyStrategy",
    "BeamSearchStrategy",
    "DiverseBeamStrategy",
    "STRATEGIES",
    "get_strategy",
]