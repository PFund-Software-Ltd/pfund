from pydantic import Field, model_validator
from pfeed.enums import DataSource
from pfund.enums import TradFiBroker
from pfund.engines.base_engine_settings import BaseEngineSettings


DEFAULT_CANCEL_ALL_AT: dict[str, bool] = {
    "start": False,
    "stop": False,
}


class TradeEngineSettings(BaseEngineSettings):
    cancel_all_at: dict[str, bool] = Field(default_factory=dict)
    # TODO: handle "broker_data_source", e.g. {'IB': 'DATABENTO'}
    broker_data_source: dict[TradFiBroker, DataSource] | None = None
    
    @model_validator(mode="after")
    def _merge_defaults(self):
        # keep any user‑provided value, fill only the missing ones
        for k, v in DEFAULT_CANCEL_ALL_AT.items():
            self.cancel_all_at.setdefault(k, v)
        return self
