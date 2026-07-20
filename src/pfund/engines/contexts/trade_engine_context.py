from typing_extensions import TypeVar

from pfund.engines.contexts.base_engine_context import BaseEngineContext
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings


SettingsT = TypeVar("SettingsT", bound=TradeEngineSettings, default=TradeEngineSettings)


class TradeEngineContext(BaseEngineContext[SettingsT]):
    pass
