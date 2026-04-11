from typing import Any

from pydantic import Field, field_serializer, field_validator, model_validator

from pfund.engines.settings.base_engine_settings import BaseEngineSettings
from pfund.utils.ray_dict import RayCompatibleDict


DEFAULT_CANCEL_ALL_AT: dict[str, bool] = {
    "start": False,
    "stop": False,
}


class TradeEngineSettings(BaseEngineSettings):
    # e.g. url='tcp://localhost'
    zmq_urls: RayCompatibleDict = Field(default_factory=RayCompatibleDict)
    zmq_ports: RayCompatibleDict = Field(default_factory=RayCompatibleDict)
    auto_stream: bool = Field(
        default=True, 
        description="""
        if False, pfund will not automatically create pfeed's data engine to stream data.
        i.e. no streaming data will be received during trading.
        Setting this to False is only useful when you have multiple trade engines or
        want to stream data manually. e.g. configure the trade engines to share the same data engine from pfeed.
        """
    )
    cancel_all_at: dict[str, bool] = Field(default_factory=dict)
    # force refetching market configs
    refetch_market_configs: bool = Field(default=False)
    # renew market configs every x days
    renew_market_configs_every_x_days: int = Field(default=7)
    # Always use the WebSocket API for actions like placing or canceling orders, even if REST is available.
    websocket_first: bool = Field(default=True)
    # Swap live data for EOD data (live data is the recorded data and EOD data is the data from a data provider)
    swap_live_for_eod: bool = Field(default=True)
    
    @model_validator(mode="after")
    def _merge_defaults(self):
        # keep any user‑provided value, fill only the missing ones
        for k, v in DEFAULT_CANCEL_ALL_AT.items():
            self.cancel_all_at.setdefault(k, v)
        return self

    @field_validator('auto_stream', mode='after')
    @classmethod
    def _warn_if_auto_stream_is_disabled(cls, v: bool) -> bool:
        from pfund_kit.style import cprint, RichColor, TextStyle
        if not v:
            cprint(
                "WARNING: 'auto_stream' is disabled, NO STREAMING DATA WILL BE RECEIVED DURING TRADING unless you manually stream data using pfeed",
                style=TextStyle.BOLD + RichColor.RED,
            )
        return v

    @field_validator('zmq_urls', 'zmq_ports', mode='before')
    @classmethod
    def _coerce_to_ray_compatible_dict(cls, v):
        if isinstance(v, RayCompatibleDict):
            return v
        return RayCompatibleDict(v or None)

    @field_serializer('zmq_urls', 'zmq_ports')
    def _serialize_ray_compatible_dict(self, v: RayCompatibleDict) -> dict[str, Any]:
        return v.to_dict()