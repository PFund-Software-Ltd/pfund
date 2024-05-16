from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.portfolios import Portfolio
    from pfund.universes import Universe
    
from abc import ABC, abstractmethod

from pfund.strategies.strategy_base import BaseStrategy


# TODO
class AllocationStrategy(BaseStrategy, ABC):
    @abstractmethod
    def allocate(self, universe: Universe, portfolio: Portfolio, *args, **kwargs):
        pass