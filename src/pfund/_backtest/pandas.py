from typing import Any

import pandas as pd

from pfund._backtest.narwhals_mixin import NarwhalsMixin


class BacktestDataFrame(NarwhalsMixin, pd.DataFrame):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Pandas doesn.*t allow columns.*"
            )
            self._open_position_inputs: dict[str, Any] = {}
            self._close_position_inputs: dict[str, Any] = {}
