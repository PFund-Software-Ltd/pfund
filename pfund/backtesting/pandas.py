import pandas as pd

from pfund.backtest_dfs.narwhals_mixin import NarwhalsMixin


class BacktestDataFrame(NarwhalsMixin, pd.DataFrame):
    pass