from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

from pfund.typing import tTradingVenue, ComponentName, EngineName


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    zmq_urls: dict[EngineName | tTradingVenue | ComponentName, str] = Field(default_factory=dict)
    zmq_ports: dict[Literal['proxy', 'router', 'publisher'] | tTradingVenue | ComponentName, int] = Field(default_factory=dict)
