from abc import ABC, abstractmethod

from pfund.strategies.allocation_strategy import AllocationStrategy


# TODO
class OptimizationStrategy(AllocationStrategy, ABC):
    @abstractmethod
    def optimize(self, universe, portfolio, *args, **kwargs):
        pass

    def allocate(self, universe, portfolio, *args, **kwargs):
        self.optimize(universe, portfolio, *args, **kwargs)