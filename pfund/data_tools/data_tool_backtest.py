'''
BacktestDataFrame class is only used to define types and improve Intellisense in IDEs,
it is NOT an actual DataFrame class.
It is referring to the actual functions defined in data_tool_xxx.py
'''
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.typing import GenericSeries


class _BacktestDataFrame:
    def create_signal(
        self, 
        buy_condition: GenericSeries | None=None,
        sell_condition: GenericSeries | None=None,
        signal: GenericSeries | None=None,
        first_only: bool=False,
    ) -> _BacktestDataFrame: ...

    def open_position(
        self,
        order_price: GenericSeries | None=None,
        order_quantity: GenericSeries | None=None,
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
        from pfeed.enums import DataTool
        from pfund.data_tools.data_tool_backtest import _BacktestDataFrame
        from pfund.engines import BacktestEngine
        data_tool = BacktestEngine.DataTool.name
        if data_tool == DataTool.pandas:
            import pandas as pd
            class PandasBacktestDataFrame(_BacktestDataFrame, pd.DataFrame):
                pass
            return PandasBacktestDataFrame
        elif data_tool == DataTool.polars:
            import polars as pl
            class PolarsBacktestDataFrame(_BacktestDataFrame, pl.DataFrame):
                pass
            return PolarsBacktestDataFrame
        # EXTEND
        else:
            raise Exception(f'Unsupported {data_tool=}')
    raise AttributeError(f"module {__name__} has no attribute {name}")
