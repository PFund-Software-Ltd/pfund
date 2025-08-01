from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund._typing import StrategyT

from pfund.strategies.strategy_base import BaseStrategy
from pfund.strategies.rebalancing_strategy import RebalancingStrategy
from pfund.strategies.diversification_strategy import DiversificationStrategy
from pfund.strategies.optimization_strategy import OptimizationStrategy
from pfund.strategies.allocation_strategy import AllocationStrategy
from pfund.utils.envs import backtest
from pfund.portfolios.portfolio import Portfolio
from pfund.universes.universe import Universe
from pfund.investment_profile import InvestmentProfile


class PortfolioStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: set_investment_profile()?
        self._profile = InvestmentProfile()
        self._universe = Universe()
        self._portfolio = Portfolio()
        self.rebalancing_strategy: RebalancingStrategy | None = None
        self.allocation_strategy: AllocationStrategy | None = None
        self.diversification_strategy: DiversificationStrategy | None = None
        self.optimization_strategy: OptimizationStrategy | None = None
    
    @property
    def universe(self) -> Universe:
        return self._universe
    
    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio
    
    @property
    def profile(self):
        return self._profile
    
    def on_start(self):
        products = self.list_products()
        positions = self.list_positions()
        balances = self.list_balances()
        self._universe.initialize(products)
        self._portfolio.initialize(positions, balances)
        return super().on_start()
    
    def add_strategy(self, strategy: StrategyT, name: str='') -> StrategyT:
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
        return super().add_strategy(strategy, name=name)
    
    # TODO: portfolio strategy is mostly not driven by event but by time
    def on_time(self):
        pass
    
    def rebalance(self, weights: dict[str, float] | None=None):
        if self.rebalancing_strategy:
            self.rebalancing_strategy.rebalance(self.universe, self.portfolio)
        else:
            # TODO: add exeuction_strategy?
            if weights:
                # TODO: rebalances current portfolio based on the weights provided
                pass
            else:
                raise ValueError('Either weights or strategy must be provided.')
    
    def diversify(self):
        if self.diversification_strategy:
            self.diversification_strategy.diversify(self.universe, self.portfolio)
        else:
            raise ValueError('Diversification strategy not provided.')
    
    def allocate(self):
        if self.allocation_strategy:
            self.allocation_strategy.allocate(self.universe, self.portfolio)
        else:
            raise ValueError('Allocation strategy not provided.')
        
    def optimize(self):
        if self.optimization_strategy:
            self.optimization_strategy.optimize(self.universe, self.portfolio)
        else:
            raise ValueError('Optimization strategy not provided.')
    
    # TODO: how to handle rebalancing? separate vectorized backtesting based on rebalancing_period?
    # TODO: also create the event-driven version
    # @backtest
    # def backtest(self):
        # long holding
        # self.df['position'] = 1
        # return self.df
