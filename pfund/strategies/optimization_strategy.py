from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.portfolio import Portfolio
    from pfund.universes.base_universe import BaseUniverse

from abc import ABC, abstractmethod

from pfund.strategies.allocation_strategy import AllocationStrategy


# TODO
class OptimizationStrategy(AllocationStrategy, ABC):
    @abstractmethod
    def optimize(self, universes: dict[str, BaseUniverse], portfolio: Portfolio, *args, **kwargs):
        pass

    def allocate(self, universes: dict[str, BaseUniverse], portfolio: Portfolio, *args, **kwargs):
        self.optimize(universes, portfolio, *args, **kwargs)