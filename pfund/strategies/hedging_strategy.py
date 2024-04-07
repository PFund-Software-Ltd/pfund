from abc import ABC, abstractmethod

from pfund.strategies.strategy_base import BaseStrategy


# TODO
class HedgingStrategy(BaseStrategy, ABC):
    @abstractmethod
    def hedge(self, portfolio, *args, **kwargs):
        pass