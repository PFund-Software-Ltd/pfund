from typing import TypeAlias

from pfund._backtest.pandas import BacktestDataFrame as BacktestPandasDataFrame
from pfund._backtest.polars import (
    BacktestDataFrame as BacktestPolarsDataFrame,
    BacktestLazyFrame as BacktestPolarsLazyFrame,
)


BacktestDataFrame: TypeAlias = BacktestPandasDataFrame | BacktestPolarsDataFrame
BacktestLazyFrame: TypeAlias = BacktestPolarsLazyFrame
