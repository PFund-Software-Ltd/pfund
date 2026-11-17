from typing import Any

from pfund.engines.contexts.base_engine_context import BaseEngineContext
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings


class TradeEngineContext(BaseEngineContext[TradeEngineSettings]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # FIXME
        self.database_storage_config = StorageConfig(
            storage=self.settings.database,
            data_path=self.settings.database_uri
            or self.config.data_path.resolve().as_uri(),
        )
