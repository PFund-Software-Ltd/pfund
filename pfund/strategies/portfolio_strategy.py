from pfund.strategies.strategy_base import BaseStrategy
from pfund.strategies.rebalancing_strategy import RebalancingStrategy
from pfund.strategies.allocation_strategy import AllocationStrategy
from pfund.strategies.diversification_strategy import DiversificationStrategy
from pfund.strategies.optimization_strategy import OptimizationStrategy


# TODO: this is the perfect example of showing strategies and sub-strategies like AllocationStrategy, RebalancingStrategy, etc.
class PortfolioStrategy(BaseStrategy):
    def rebalance(self, weights: dict[str, float] | None=None, strategy: RebalancingStrategy | None=None):
        if weights:
            # TODO: rebalances current portfolio based on the weights provided
            pass
        elif strategy:
            strategy.rebalance(self.portfolio)
        else:
            raise ValueError('Either weights or strategy must be provided.')
    
    def allocate(self, strategy: AllocationStrategy | DiversificationStrategy | OptimizationStrategy):
        strategy.allocate(self.universe, self.portfolio)
    
    def backtest(self):
        # long holding
        self.df['position'] = 1
        # TODO: how to handle rebalancing? separate vectorized backtesting based on rebalancing_period?
        # TODO: also create the event-driven version
        return self.df
