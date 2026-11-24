from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:
    import polars as pl

    from pfund.backtest._mixins.portfolio_backtest_mixin import PortfolioBacktestMixin
    from pfund.backtest._mixins.product_backtest_mixin import ProductBacktestMixin

    class BacktestDataFrame(
        ProductBacktestMixin, PortfolioBacktestMixin, pl.DataFrame
    ):
        """Annotation-only type: at runtime the df is a plain native DataFrame
        whose class has the backtest methods attached by _setup_backtest_df_for_vectorized_mode()."""

        # Both mixins define backtest() with different signatures; at runtime only
        # one is attached (by create_signal() or create_weight()). This merged stub
        # accepts both call shapes: product mode takes no args, portfolio mode uses them.
        def backtest(
            self,
            initial_capital: float = 1_000_000,
            compound: bool = True,
            check: bool = True,
        ) -> Self: ...


from pfund.components.actor_proxy import ActorProxy
from pfund.components.features.feature_base import BaseFeature
from pfund.components.models.model_base import BaseModel
from pfund.components.strategies.strategy_base import BaseStrategy

type Component = BaseStrategy | BaseModel | BaseFeature
type ComponentProxy = (
    ActorProxy[BaseStrategy] | ActorProxy[BaseModel] | ActorProxy[BaseFeature]
)
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
