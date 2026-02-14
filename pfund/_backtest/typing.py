from typing import TypeAlias

from pfund._backtest.pandas import BacktestDataFrame as BacktestPandasDataFrame
from pfund._backtest.polars import BacktestDataFrame as BacktestPolarsDataFrame


BacktestDataFrame: TypeAlias = BacktestPandasDataFrame | BacktestPolarsDataFrame


__all__ = [
    'BacktestDataFrame',
    'BacktestPandasDataFrame',
    'BacktestPolarsDataFrame',
]