from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sklearn.model_selection import TimeSeriesSplit
    from pfeed.storages.storage_config import StorageConfig

    from pfund.engines.base_engine import DataRangeDict
    from pfund._backtest.cv.base import CrossValidatorDatasetPeriods
    from pfund._backtest.dataset_splitter import DatasetPeriods, DatasetSplitsDict

from pfund_kit.style import RichColor, TextStyle, cprint

from pfund.datas.resolution import Resolution
from pfund.enums import Environment, BacktestMode
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.engines.contexts.base_engine_context import BaseEngineContext
from pfund._backtest.dataset_splitter import DatasetSplitter


class BacktestEngineContext(BaseEngineContext[BacktestEngineSettings]):
    def __init__(
        self,
        env: Environment,
        name: str,
        data_range: str | Resolution | DataRangeDict | tuple[str, str] | Literal["ytd"],
        settings: BacktestEngineSettings | None = None,
        storage_config: StorageConfig | None = None,
        mode: BacktestMode
        | Literal["vectorized", "event_driven"] = BacktestMode.VECTORIZED,
        dataset_splits: int | DatasetSplitsDict | TimeSeriesSplit = 721,
    ):
        super().__init__(
            env=env,
            name=name,
            data_range=data_range,
            settings=settings,
            storage_config=storage_config,
        )
        self.mode = BacktestMode[mode.upper()]
        if self.mode == BacktestMode.EVENT_DRIVEN and self.settings.reuse_signals:
            cprint(
                "Warning: Reusing pre-computed features to speed up event-driven backtesting,\n"
                + "i.e. computing features on the fly will be skipped",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
        assert self.data_start is not None
        assert self.data_end is not None
        self.dataset_splitter = DatasetSplitter(
            dataset_start=self.data_start,
            dataset_end=self.data_end,
            dataset_splits=dataset_splits,
        )

    @property
    def dataset_periods(self) -> DatasetPeriods | list[CrossValidatorDatasetPeriods]:
        return self.dataset_splitter.dataset_periods
