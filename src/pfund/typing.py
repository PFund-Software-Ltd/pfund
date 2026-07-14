from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar, Any

if TYPE_CHECKING:
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.entities.products.product_base import BaseProduct
    from pfund.entities.accounts.account_base import BaseAccount


Component: TypeAlias = "BaseStrategy | BaseModel | BaseFeature"
ComponentT = TypeVar("ComponentT", bound="Component")
StrategyT = TypeVar("StrategyT", bound="BaseStrategy")
ModelT = TypeVar("ModelT", bound="BaseModel")
FeatureT = TypeVar("FeatureT", bound="BaseFeature")
ProductT = TypeVar("ProductT", bound="BaseProduct")
AccountT = TypeVar("AccountT", bound="BaseAccount")


EngineName: TypeAlias = str
ComponentName: TypeAlias = str
ProductName: TypeAlias = str
AccountName: TypeAlias = str
ColumnName: TypeAlias = str
Currency: TypeAlias = str
FullDataChannel: TypeAlias = str
Signals: TypeAlias = dict[ColumnName, Any]


EngineNameWithProxy: TypeAlias = EngineName
ComponentNameWithData: TypeAlias = ComponentName
ComponentNameWithLogger: TypeAlias = ComponentName
ZMQUrlKey = EngineName | Literal["data_engine"] | ComponentName
# NOTE: only senders need to set ports
ZMQPortKey = (
    ZMQUrlKey
    | ComponentNameWithData  # {component_name}_data
    | ComponentNameWithLogger  # {component_name}_logger
)
