from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, ConfigDict

from pfund.typing import tTradingVenue, ComponentName, EngineName


ComponentNameWithData: TypeAlias = str


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    zmq_urls: dict[EngineName | tTradingVenue | ComponentName, str] = Field(default_factory=dict)
    zmq_ports: dict[
        Literal['proxy', 'router', 'publisher'] | 
        tTradingVenue | 
        # each component has TWO ZeroMQ ports:
        # ComponentName is used for component's signals_zmq
        # ComponentNameWithData is used for component's data_zmq
        ComponentName |
        ComponentNameWithData  # {component_name}_data
    , int] = Field(default_factory=dict)
