from pydantic import Field, model_validator
from pfund.engines.base_engine_settings import BaseEngineSettings


DEFAULT_CANCEL_ALL_AT: dict[str, bool] = {
    "start": False,
    "stop": False,
}


class TradeEngineSettings(BaseEngineSettings):
    cancel_all_at: dict[str, bool] = Field(default_factory=dict)
    # force refetching market configs
    refetch_market_configs: bool = Field(default=False)
    # renew market configs every x days
    renew_market_configs_every_x_days: int = Field(default=7)
    # Always use the WebSocket API for actions like placing or canceling orders, even if REST is available.
    websocket_first: bool = Field(default=True)
    
    @model_validator(mode="after")
    def _merge_defaults(self):
        # keep any user‑provided value, fill only the missing ones
        for k, v in DEFAULT_CANCEL_ALL_AT.items():
            self.cancel_all_at.setdefault(k, v)
        return self
