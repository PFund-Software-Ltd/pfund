from pydantic import BaseModel, Field, ConfigDict

from pfund._typing import tTradingVenue, ComponentName, EngineName, ZeroMQName




class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # e.g. url='tcp://localhost'
    zmq_urls: dict[EngineName | tTradingVenue | ComponentName, str] = Field(default_factory=dict)
    zmq_ports: dict[ZeroMQName, int] = Field(default_factory=dict)
