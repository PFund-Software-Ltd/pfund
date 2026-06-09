from typing import Any

import polars as pl

from pfund._backtest.narwhals_mixin import NarwhalsMixin

# TODO: test on engine="gpu"
# pl.Config.set_engine_affinity(engine="streaming")


# TODO: maybe create a subclass like SafeFrame(pd.DataFrame) to prevent users from peeking into the future?
# e.g. df['close'] = df['close'].shift(-1) should not be allowed
class BacktestDataFrame(NarwhalsMixin, pl.DataFrame):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._open_position_inputs: dict[str, Any] = {}
        self._close_position_inputs: dict[str, Any] = {}
