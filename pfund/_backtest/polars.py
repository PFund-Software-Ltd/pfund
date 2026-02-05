from __future__ import annotations

import polars as pl

from pfund._backtest.narwhals_mixin import NarwhalsMixin


class BacktestDataFrame(NarwhalsMixin, pl.DataFrame):
    def lazy(self) -> BacktestLazyFrame:
        """Convert to lazy evaluation."""
        return BacktestLazyFrame(super().lazy())


class BacktestLazyFrame(NarwhalsMixin, pl.LazyFrame):
    def collect(self) -> BacktestDataFrame:
        """Execute and return eager DataFrame."""
        return BacktestDataFrame(super().collect())