from typing import Any

from typing_extensions import TypeVar
from pfeed.storages.storage_config import StorageConfig

from pfund.engines.contexts.base_engine_context import BaseEngineContext
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings


SettingsT = TypeVar("SettingsT", bound=TradeEngineSettings, default=TradeEngineSettings)


class TradeEngineContext(BaseEngineContext[SettingsT]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.database_storage_config = StorageConfig(
            storage=self.settings.database,
            data_path=self.settings.database_path,
        )
