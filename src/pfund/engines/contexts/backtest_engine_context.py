from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pfund.backtest.cv.dataset_splits import DatasetSplitsInput, FoldNum, SplitPart
    from pfund.engines.base_engine import DataRangeDict

import os
import warnings

from pfund_blocks.enums.env import Environment
from pfund_blocks.resolution import Resolution

from pfund.backtest.cv.dataset_splits import DatasetSplits
from pfund.backtest.cv.holdout import Holdout
from pfund.engines.contexts.base_engine_context import BaseEngineContext
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.enums import BacktestMode


class BacktestEngineContext(BaseEngineContext[BacktestEngineSettings]):
    def __init__(
        self,
        *,
        env: Environment,
        name: str,
        data_range: str | Resolution | DataRangeDict | tuple[str, str] | Literal["ytd"],
        settings: BacktestEngineSettings | None = None,
        mode: BacktestMode
        | Literal["vectorized", "event_driven"] = BacktestMode.VECTORIZED,
        dataset_splits: DatasetSplitsInput | None = None,
    ):
        super().__init__(
            env=env,
            name=name,
            data_range=data_range,
            settings=settings,
        )
        self.mode = BacktestMode[mode.upper()]

        # TEMP
        if self.mode == BacktestMode.EVENT_DRIVEN:
            raise ValueError("Event-driven backtesting is not supported yet")

        if self.mode == BacktestMode.EVENT_DRIVEN and self.settings.reuse_signals:
            import pfund_kit

            import pfund

            warnings.warn(
                "Reusing pre-computed signals to speed up event-driven backtesting, "
                + "i.e. computing signals on the fly will be skipped. "
                + "Set reuse_signals=False in the settings to compute them on the fly.",
                UserWarning,
                # point at the user's engine construction, past EngineMeta/SingletonMeta and __init__ frames
                skip_file_prefixes=(
                    os.path.dirname(pfund.__file__),
                    os.path.dirname(pfund_kit.__file__),
                ),
            )
        # None means 100% train, so there's always a split to record what a component ran on
        assert self.data_start is not None and self.data_end is not None
        self.dataset_splits = DatasetSplits(
            Holdout(train=1.0, val=0.0, test=0.0)
            if dataset_splits is None
            else dataset_splits,
            data_start=self.data_start,
            data_end=self.data_end,
            split_resolution=self.settings.split_resolution,
        )
        self._active_split: tuple[SplitPart, FoldNum | None] | None = None

    def set_active_split(self, split: tuple[SplitPart, FoldNum | None] | None):
        self._active_split = split

    @property
    def active_split(self) -> tuple[SplitPart, FoldNum | None]:
        if self._active_split is None:
            raise RuntimeError(
                "No dataset split is active. Call engine.use_split() to select 'train', 'val' or 'test"
            )
        return self._active_split
