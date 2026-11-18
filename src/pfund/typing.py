from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel
    from pfund.components.strategies.strategy_base import BaseStrategy


type Component = BaseStrategy | BaseModel | BaseFeature
ComponentT = TypeVar("ComponentT", bound="Component")
StrategyT = TypeVar("StrategyT", bound="BaseStrategy")
ModelT = TypeVar("ModelT", bound="BaseModel")
FeatureT = TypeVar("FeatureT", bound="BaseFeature")


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
