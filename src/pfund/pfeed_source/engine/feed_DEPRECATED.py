from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    from pfund.enums import ComponentType

import datetime

from pfeed.enums import DataCategory
from pfeed.base.feed import BaseFeed
from pfeed.sources.pfund.engine_data_model import PFundEngineDataModel
from pfeed.sources.pfund.source import PFundSource
from pfeed.storages.storage_config import StorageConfig
from pfund.engines.contexts.base_engine_context import BaseEngineContext
from pfund_blocks.enums.env import Environment


# TODO: get around the data source issue, since technically its data source is the engine.
# this feed should be able to get backtesting data from pfund's BacktestEngine for monitoring and analysis puporse
# some functions require api calls (e.g. get dynamic backtest results) and some do not (e.g. load backtest hisory)
# TODO: need to specify which engine if there are multiple engines running
class PFundEngineFeed(BaseFeed):
    DataSource: ClassVar[type[PFundSource]] = PFundSource
    data_source: PFundSource

    DataModel: ClassVar[type[PFundEngineDataModel]] = PFundEngineDataModel
    data_domain: ClassVar[DataCategory] = DataCategory.ENGINE_DATA

    # def __init__(
    #     self,
    #     pipeline_mode: bool = False,
    #     num_workers: int | None = None,
    # ):
    #     super().__init__(pipeline_mode=pipeline_mode, num_workers=num_workers)

    def transform(self, *funcs: Callable[..., Any]) -> Self:
        raise NotImplementedError(
            f"{self.name} does not support transform(): artifacts are persisted as-is"
        )

    def create_data_model(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        component_name: str | None = None,
        component_type: ComponentType | None = None,
    ) -> PFundEngineDataModel:
        return PFundEngineDataModel(...)

    def _create_data_model_from_request(self, request):
        pass

    def download(
        self,
        engine_name: str = "engine",
        env: Environment | str = Environment.BACKTEST,
        project_name: str = BaseEngineContext.DEFAULT_PROJECT_NAME,
        run_name: str = BaseEngineContext.DEFAULT_RUN_NAME,
        storage_config: StorageConfig | None = None,
    ):
        pass

    def _download_impl(self, data_model: PFundEngineDataModel):
        pass

    def _get_default_transformations_for_download(self, *args, **kwargs):
        pass

    def retrieve(self, engine_name: str = "engine"):
        pass

    def _retrieve_impl(self, data_model: PFundEngineDataModel):
        pass

    def _get_default_transformations_for_retrieve(self, *args, **kwargs):
        pass

    # TODO: stream engine's states, connect to mtflow's ws server
    def stream(
        self, engine_name: str = "engine", env: Environment | str = Environment.BACKTEST
    ):
        pass

    def _stream_impl(self, data_model: PFundEngineDataModel):
        pass

    def _create_batch_dataflows(self, *args, **kwargs):
        pass

    def run(self, use_prefect: bool = False, **prefect_kwargs: Any):
        pass
