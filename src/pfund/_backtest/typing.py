import pandas as pd
import polars as pl

from pfund._backtest.portfolio_backtest_mixin import PortfolioBacktestMixin
from pfund._backtest.product_backtest_mixin import ProductBacktestMixin


class PolarsBacktestDataFrame(
    ProductBacktestMixin, PortfolioBacktestMixin, pl.DataFrame
):
    """Annotation-only type: at runtime the df is a plain native DataFrame
    whose class has the backtest methods attached by _setup_backtest_df_for_fast_mode()."""


class PandasBacktestDataFrame(
    ProductBacktestMixin, PortfolioBacktestMixin, pd.DataFrame
):
    """Annotation-only type: at runtime the df is a plain native DataFrame
    whose class has the backtest methods attached by _setup_backtest_df_for_fast_mode()."""
