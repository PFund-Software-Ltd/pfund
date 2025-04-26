from mtflow.typing import tZMQ_MESSENGER
from pfund.typing import tTRADING_VENUE, ComponentName, EngineName
from pydantic import BaseModel, Field, model_validator


DEFAULT_CANCEL_ALL_AT: dict[str, bool] = {
    "start": True,
    "stop": True,
}


# TODO: zmq_urls, zmq_ports should be moved into BaseEngineSettings? 
# since they are also used in backtesting
class TradeEngineSettings(BaseModel):
    zmq_urls: dict[EngineName | tTRADING_VENUE | ComponentName, str] = Field(default_factory=dict)
    zmq_ports: dict[tZMQ_MESSENGER | tTRADING_VENUE | ComponentName, int] = Field(default_factory=dict)
    cancel_all_at: dict[str, bool] = Field(default_factory=dict)
    
    @model_validator(mode="after")
    def _merge_defaults(self):
        # keep any user‑provided value, fill only the missing ones
        for k, v in DEFAULT_CANCEL_ALL_AT.items():
            self.cancel_all_at.setdefault(k, v)
        return self
