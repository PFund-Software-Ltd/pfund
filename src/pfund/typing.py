from __future__ import annotations

from typing import Any, Literal

from pfund.components.features.feature_base import BaseFeature
from pfund.components.models.model_base import BaseModel
from pfund.components.strategies.strategy_base import BaseStrategy

type Component = BaseStrategy | BaseModel | BaseFeature
type EngineName = str
type ComponentName = str
type ColumnName = str
type Signals = dict[ColumnName, Any]


type EngineNameWithProxy = EngineName
type ComponentNameWithData = ComponentName
type ComponentNameWithLogger = ComponentName
type ZMQUrlKey = EngineName | Literal["data_engine"] | ComponentName
# NOTE: only senders need to set ports
type ZMQPortKey = (
    ZMQUrlKey
    | ComponentNameWithData  # {component_name}_data
    | ComponentNameWithLogger  # {component_name}_logger
)
