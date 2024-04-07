from abc import ABC, abstractmethod

from pfund.strategies.strategy_base import BaseStrategy


# TODO
class RebalancingStrategy(BaseStrategy, ABC):
    @abstractmethod
    def rebalance(self, portfolio, *args, **kwargs):
        pass