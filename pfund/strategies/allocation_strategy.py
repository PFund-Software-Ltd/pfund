from abc import ABC, abstractmethod

from pfund.strategies.strategy_base import BaseStrategy


# TODO
class AllocationStrategy(BaseStrategy, ABC):
    @abstractmethod
    def allocate(self, universe, portfolio, *args, **kwargs):
        pass