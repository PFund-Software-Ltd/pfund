'''
BacktestDataFrame class is only used to define types and improve Intellisense in IDEs,
it is NOT an actual DataFrame class.
It is referring to the actual functions defined in data_tool_xxx.py
'''
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.typing.core import tSeries


class _BacktestDataFrame:
    def create_signal(
        self, 
        buy_condition: tSeries | None=None,
        sell_condition: tSeries | None=None,
        signal: tSeries | None=None,
        first_only: bool=False,
    ) -> _BacktestDataFrame: ...

    def open_position(
        self,
        order_price: tSeries | None=None,
        order_quantity: tSeries | None=None,
        first_only: bool=True,
        long_only: bool=False,
        short_only: bool=False,
        ignore_sizing: bool=True,
        fill_ratio: float=0.1,
    ) -> _BacktestDataFrame: ...
    
    def close_position(
        self,
        take_profit: float | None=None,
        stop_loss: float | None=None,
    ) -> _BacktestDataFrame: ...

        
def __getattr__(name):
    if name == 'BacktestDataFrame':
        from pfund.utils.utils import get_engine_class
        from pfund.typing.data_tool import _BacktestDataFrame
        from pfeed.const.enums import DataTool
        Engine = get_engine_class()
        if Engine.data_tool == DataTool.PANDAS:
            import pandas as pd
            class PandasBacktestDataFrame(_BacktestDataFrame, pd.DataFrame):
                pass
            return PandasBacktestDataFrame
        elif Engine.data_tool == DataTool.POLARS:
            import polars as pl
            class PolarsBacktestDataFrame(_BacktestDataFrame, pl.DataFrame):
                pass
            return PolarsBacktestDataFrame
        # EXTEND
        else:
            raise Exception(f'Unsupported data tool: {Engine.data_tool.value}')
    raise AttributeError(f"module {__name__} has no attribute {name}")
