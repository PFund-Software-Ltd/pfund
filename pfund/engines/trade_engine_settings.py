from pydantic import Field, model_validator
from pfund.engines.base_engine_settings import BaseEngineSettings


DEFAULT_CANCEL_ALL_AT: dict[str, bool] = {
    "start": True,
    "stop": True,
}


class TradeEngineSettings(BaseEngineSettings):
    cancel_all_at: dict[str, bool] = Field(default_factory=dict)
    
    @model_validator(mode="after")
    def _merge_defaults(self):
        # keep any user‑provided value, fill only the missing ones
        for k, v in DEFAULT_CANCEL_ALL_AT.items():
            self.cancel_all_at.setdefault(k, v)
        return self
