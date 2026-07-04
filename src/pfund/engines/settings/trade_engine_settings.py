from typing import Any

from pydantic import Field, field_serializer, field_validator

from pfund.engines.settings.base_engine_settings import BaseEngineSettings
from pfund.utils.ray_dict import RayCompatibleDict


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
        """,
    )
    swap_live_for_eod: bool = Field(
        default=True,
        description="When True, replace recorded live data with the provider's cleaned end-of-day data once available.",
    )

    @field_validator("auto_stream", mode="after")
    @classmethod
    def _warn_if_auto_stream_is_disabled(cls, v: bool) -> bool:
        from pfund_kit.style import RichColor, TextStyle, cprint

        if not v:
            cprint(
                "WARNING: 'auto_stream' is disabled, NO STREAMING DATA WILL BE RECEIVED DURING TRADING unless you manually stream data using pfeed",
                style=TextStyle.BOLD + RichColor.RED,
            )
        return v

    @field_validator("zmq_urls", "zmq_ports", mode="before")
    @classmethod
    def _coerce_to_ray_compatible_dict(cls, v):
        if isinstance(v, RayCompatibleDict):
            return v
        return RayCompatibleDict(v or None)

    @field_serializer("zmq_urls", "zmq_ports")
    def _serialize_ray_compatible_dict(self, v: RayCompatibleDict) -> dict[str, Any]:
        return v.to_dict()
