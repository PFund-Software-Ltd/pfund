import pandas as pd

from pfund._backtest.narwhals_mixin import NarwhalsMixin


class BacktestDataFrame(NarwhalsMixin, pd.DataFrame):
    pass