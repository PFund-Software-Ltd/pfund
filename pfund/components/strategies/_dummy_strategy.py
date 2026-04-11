# pyright: reportUnknownParameterType=false
from typing import Any

from pfund.components.strategies.strategy_base import BaseStrategy


class DummyStrategy(BaseStrategy):
    name: str = '_dummy'
    
    # add event driven functions to dummy strategy to avoid NotImplementedError in backtesting
    def on_quote(self, *args: Any, **kwargs: Any):
        pass
    
    def on_tick(self, *args: Any, **kwargs: Any):
        pass

    def on_bar(self, *args: Any, **kwargs: Any):
        pass
