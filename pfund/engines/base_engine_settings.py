from pydantic import BaseModel, Field, ConfigDict

from mtflow.typing import tZMQ_MESSENGER
from pfund.typing import tTRADING_VENUE, ComponentName, EngineName


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    zmq_urls: dict[EngineName | tTRADING_VENUE | ComponentName, str] = Field(default_factory=dict)
    zmq_ports: dict[tZMQ_MESSENGER | tTRADING_VENUE | ComponentName, int] = Field(default_factory=dict)
