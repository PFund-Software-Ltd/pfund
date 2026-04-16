# pyright: reportUnknownParameterType=false
from __future__ import annotations
from typing import Any

from pfund.components.strategies.strategy_base import BaseStrategy


class _DummyStrategy(BaseStrategy):  # pyright: ignore[reportUnusedClass]
    # add event driven functions to dummy strategy to avoid NotImplementedError in backtesting
    def on_quote(self, *args: Any, **kwargs: Any):
        pass
    
    def on_tick(self, *args: Any, **kwargs: Any):
        pass

    def on_bar(self, *args: Any, **kwargs: Any):
        pass

    def trade(self, *args: Any, **kwargs: Any):
        pass
