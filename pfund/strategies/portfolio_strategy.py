from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.core import tStrategy
    from pfund.brokers.broker_base import BaseBroker
    from pfund.universes.base_universe import BaseUniverse

from pfund.strategies.strategy_base import BaseStrategy
from pfund.strategies.rebalancing_strategy import RebalancingStrategy
from pfund.strategies.diversification_strategy import DiversificationStrategy
from pfund.strategies.optimization_strategy import OptimizationStrategy
from pfund.strategies.allocation_strategy import AllocationStrategy
from pfund.utils.envs import backtest
from pfund.portfolio import Portfolio
from pfund.universes import TradingUniverse, CryptoUniverse, DefiUniverse


class PortfolioStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: set_investment_profile()?
        self._investment_profile = None
        self._universes = {}  # {bkr: universe}
        self._portfolio = Portfolio()
        self.rebalancing_strategy: RebalancingStrategy | None = None
        self.allocation_strategy: AllocationStrategy | None = None
        self.diversification_strategy: DiversificationStrategy | None = None
        self.optimization_strategy: OptimizationStrategy | None = None
        
    @property
    def universes(self) -> dict[str, BaseUniverse]:
        return self._universes
    
    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio
    
    @property
    def investment_profile(self):
        return self._investment_profile
    
    def add_strategy(self, strategy: tStrategy, name: str='', is_parallel=False) -> tStrategy:
        # NOTE: if there are multiple strategies of the same type, only the first one will be used as the default strategy
        if isinstance(strategy, RebalancingStrategy):
            if not self.rebalancing_strategy:
                self.rebalancing_strategy = strategy
        elif isinstance(strategy, DiversificationStrategy):
            if not self.diversification_strategy:
                self.diversification_strategy = strategy
        elif isinstance(strategy, OptimizationStrategy):
            if not self.optimization_strategy:
                self.optimization_strategy = strategy
        elif isinstance(strategy, AllocationStrategy):
            if not self.allocation_strategy:
                self.allocation_strategy = strategy
        return super().add_strategy(strategy, name=name, is_parallel=is_parallel)
    
    def get_universe(self, bkr: str) -> BaseUniverse:
        return self.universes[bkr]
    
    def add_universe(self, bkr: str):
        bkr = bkr.upper()
        if bkr == 'CRYPTO':
            self.universes[bkr] = CryptoUniverse()
        elif bkr == 'DEFI':
            self.universes[bkr] = DefiUniverse()
        else:
            self.universes[bkr] = TradingUniverse()
    
    def add_broker(self, bkr: str) -> BaseBroker:
        self.add_universe(bkr)
        return super().add_broker(bkr)
    
    # TODO: portfolio strategy is mostly not driven by event but by time
    def on_time(self):
        pass
    
    def rebalance(self, weights: dict[str, float] | None=None):
        if self.rebalancing_strategy:
            self.rebalancing_strategy.rebalance(self.universes, self.portfolio)
        else:
            if weights:
                # TODO: rebalances current portfolio based on the weights provided
                pass
            else:
                raise ValueError('Either weights or strategy must be provided.')
    
    def diversify(self):
        if self.diversification_strategy:
            self.diversification_strategy.diversify(self.universes, self.portfolio)
        else:
            raise ValueError('Diversification strategy not provided.')
    
    def allocate(self):
        if self.allocation_strategy:
            self.allocation_strategy.allocate(self.universes, self.portfolio)
        else:
            raise ValueError('Allocation strategy not provided.')
        
    def optimize(self):
        if self.optimization_strategy:
            self.optimization_strategy.optimize(self.universes, self.portfolio)
        else:
            raise ValueError('Optimization strategy not provided.')
    
    @backtest
    def backtest(self):
        # long holding
        self.df['position'] = 1
        # TODO: how to handle rebalancing? separate vectorized backtesting based on rebalancing_period?
        # TODO: also create the event-driven version
        return self.df
