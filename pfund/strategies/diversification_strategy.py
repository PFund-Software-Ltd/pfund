from abc import ABC, abstractmethod

from pfund.strategies.allocation_strategy import AllocationStrategy


# TODO
class DiversificationStrategy(AllocationStrategy, ABC):
    @abstractmethod
    def diversify(self, universe, portfolio, *args, **kwargs):
        pass
    
    def allocate(self, universe, portfolio, *args, **kwargs):
        self.diversify(universe, portfolio, *args, **kwargs)