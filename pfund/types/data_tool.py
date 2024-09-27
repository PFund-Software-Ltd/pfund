'''
BacktestDataFrame class is only used to define types and improve Intellisense in IDEs,
it is NOT an actual DataFrame class.
It is referring to the actual functions defined in data_tool_xxx.py
'''
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.types.core import tSeries
 

class _BacktestDataFrame:
    def create_signal(
        self, 
        product: str | None=None,
        buy_condition: tSeries | None=None,
        sell_condition: tSeries | None=None,
        signal: tSeries | None=None,
        is_nan_signal: bool=False,
        first_only: bool=False,
    ) -> _BacktestDataFrame: ...

    def open_position(
        self,
        product: str | None=None,
        order_price: tSeries | None=None,
        order_quantity: tSeries | None=None,
        first_only: bool=True,
        ignore_sizing: bool=False,
        long_only: bool=False,
        short_only: bool=False,
    ) -> _BacktestDataFrame: ...
    
    def close_position(
        self,
        product: str | None=None,
        for_loop: bool=False,
        take_profit: float | None=None,
        stop_loss: float | None=None,
        time_window: int | None=None,
    ) -> _BacktestDataFrame: ...

        
    create = create_signal
    open = open_position
    close = close_position


def __getattr__(name):
    if name == 'BacktestDataFrame':
        from pfund.utils.utils import get_engine_class
        from pfund.types.data_tool import _BacktestDataFrame
        Engine = get_engine_class()
        if Engine.data_tool == 'pandas':
            import pandas as pd
            class PandasBacktestDataFrame(_BacktestDataFrame, pd.DataFrame):
                pass
            return PandasBacktestDataFrame
        elif Engine.data_tool == 'polars':
            import polars as pl
            class PolarsBacktestDataFrame(_BacktestDataFrame, pl.DataFrame):
                pass
            return PolarsBacktestDataFrame
        # EXTEND
        else:
            raise Exception(f'Unsupported data tool: {Engine.data_tool}')
    raise AttributeError(f"module {__name__} has no attribute {name}")
