from pydantic import BaseModel, Field, ConfigDict

from pfund._typing import ComponentName, EngineName, ZeroMQSenderName


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # e.g. url='tcp://localhost'
    zmq_urls: dict[EngineName | ComponentName, str] = Field(default_factory=dict)
    zmq_ports: dict[ZeroMQSenderName, int] = Field(default_factory=dict)
