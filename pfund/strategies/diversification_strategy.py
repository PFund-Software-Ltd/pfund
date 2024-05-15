from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.portfolio import Portfolio
    from pfund.universes.base_universe import BaseUniverse
    
from abc import ABC, abstractmethod

from pfund.strategies.allocation_strategy import AllocationStrategy


# TODO
class DiversificationStrategy(AllocationStrategy, ABC):
    @abstractmethod
    def diversify(self, universes: dict[str, BaseUniverse], portfolio: Portfolio, *args, **kwargs):
        pass
    
    def allocate(self, universes: dict[str, BaseUniverse], portfolio: Portfolio, *args, **kwargs):
        self.diversify(universes, portfolio, *args, **kwargs)