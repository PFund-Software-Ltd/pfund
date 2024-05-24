from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.portfolios import Portfolio
    from pfund.universes import Universe
    from pfund.investment_profile import InvestmentProfile

from abc import ABC, abstractmethod

from pfund.strategies.strategy_base import BaseStrategy


# TODO
class HedgingStrategy(BaseStrategy, ABC):
    @abstractmethod
    def hedge(self, universe: Universe, portfolio: Portfolio, profile: InvestmentProfile, *args, **kwargs):
        pass