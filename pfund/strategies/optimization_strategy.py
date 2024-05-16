from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.portfolios import Portfolio
    from pfund.universes import Universe
    from pfund.investment_profile import InvestmentProfile

from abc import ABC, abstractmethod

from pfund.strategies.allocation_strategy import AllocationStrategy


# TODO
class OptimizationStrategy(AllocationStrategy, ABC):
    @abstractmethod
    def optimize(self, universe: Universe, portfolio: Portfolio, profile: InvestmentProfile, *args, **kwargs):
        pass

    def allocate(self, universe: Universe, portfolio: Portfolio, profile: InvestmentProfile, *args, **kwargs):
        self.optimize(universe, portfolio, profile, *args, **kwargs)