from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from pfund.components.actor_proxy import ActorProxy
    from pfund.entities.accounts.account_base import BaseAccount
    from pfund.entities.balances.balance_base import BaseBalance
    from pfund.entities.orders.order_base import BaseOrder
    from pfund.entities.positions.position_base import BasePosition
    from pfund.entities.products.product_base import BaseProduct
    from pfund.engines.trade_engine import TradeEngine
    from pfund.typing import (
        AccountName,
        Component,
        ComponentName,
        StrategyT,
        Signals,
    )

from abc import ABC, abstractmethod

import narwhals as nw

from pfund.components.mixin import ComponentMixin
from pfund.components.strategies.strategy_meta import MetaStrategy
from pfund.managers import OrderManager, PortfolioManager, RiskManager
from pfund.enums import ComponentType


class BaseStrategy(ComponentMixin, ABC, metaclass=MetaStrategy):
    def __init__(self, *args: Any, **kwargs: Any):
        self.component_type = ComponentType.strategy
        self._df_form: Literal["wide", "long"] = "long"
        self.accounts: dict[AccountName, BaseAccount] = {}
        self.strategies: dict[str, BaseStrategy] = {}
        self._order_manager = OrderManager()
        self._portfolio_manager = PortfolioManager()
        self._risk_manager = RiskManager()
        # In-process handle to the engine, injected by TradeEngine._setup() ONLY on the
        # pure-local path (not self._is_using_zmq()). Stays None when ZMQ/Ray is in use
        self._engine: TradeEngine | None = None
        self.__mixin_post_init__(
            *args, **kwargs
        )  # calls ComponentMixin.__mixin_post_init__()

    @abstractmethod
    def decide(self, X: IntoDataFrame) -> list[BaseOrder]:
        pass

    @property
    def order_manager(self) -> OrderManager:
        return self._order_manager

    om = order_manager

    @property
    def portfolio_manager(self) -> PortfolioManager:
        return self._portfolio_manager

    pm = portfolio_manager

    @property
    def risk_manager(self) -> RiskManager:
        return self._risk_manager

    rm = risk_manager

    # TODO: load strategy's signal_df from parquet
    def load(self):
        pass

    # TODO: dump strategy's signal_df to parquet
    def dump(self):
        pass

    # TODO:
    def signalize(self, X: IntoDataFrame) -> Signals:
        """Creates signals of this component

        Args:
            X: features df

        Returns:
            dict[ColumnName, Any]: The predicted signals.
        """
        signals = self.decide(X)

    # TODO: {product.name}_signal
    def _get_default_signal_cols(self, num_cols: int) -> list[str]:
        pass

    def get_component(
        self, name: ComponentName
    ) -> Component | ActorProxy[Component] | None:
        return self.strategies.get(name, None) or super().get_component(name)

    def get_components(self) -> list[Component | ActorProxy[Component]]:
        return [*self.strategies.values(), *super().get_components()]

    def get_position(self, account: BaseAccount, pdt: str) -> BasePosition | None:
        return self.positions[account].get(pdt, None)

    def get_balance(self, account: BaseAccount, ccy: str) -> BaseBalance | None:
        return self.balances[account].get(ccy, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "strategies": list(self.strategies),
        }

    def get_accounts(self) -> list[BaseAccount]:
        return list(self.accounts.values())

    def add_account(self, account: BaseAccount) -> BaseAccount:
        if not self.is_top_component():
            raise ValueError(f"Sub-strategy '{self.name}' cannot add accounts")
        if account.env != self.env:
            raise ValueError(
                f"account env {account.env} does not match strategy env {self.env}"
            )
        self.accounts[account.name] = account
        self.logger.debug(f'added {account.venue} account "{account.name}"')
        return account

    # FIXME: update, refer to or reuse _add_component() in component_mixin.py (rmb to update _add_component in backtest mixin)
    # TODO: _setup_logging, _set_engine etc.
    # TODO: make sure sub-strategy's df_form is the same as the top strategy's df_form
    def add_strategy(self, strategy: StrategyT, name: str = "") -> StrategyT:
        if strategy.is_top_component():
            raise ValueError(
                f"Top strategy '{self.name}' cannot be added as a sub-strategy"
            )
        assert isinstance(strategy, BaseStrategy), (
            f"strategy '{strategy.__class__.__name__}' is not an instance of BaseStrategy. Please create your strategy using 'class {strategy.__class__.__name__}(pf.Strategy)'"
        )
        if name:
            strategy._set_name(name)
        strategy._set_trading_store()
        strategy._create_logger()
        strategy._set_resolution(self.resolution)
        strat = strategy.name
        if strat in self.strategies:
            return self.strategies[strat]
        self.strategies[strat] = strategy
        self.logger.debug(f"added sub-strategy '{strat}'")
        return strategy

    def _on_update(self, update: Any):
        """Transport-agnostic sink for venue updates → this strategy's managers.

        Called with a fully-formed update object from BOTH paths: the engine's local
        in-process delivery, and databoy's _data_zmq recv (after reconstructing from the
        published dict via model_validate). Mirrors the engine's _run_updates_loop match,
        one level down into the strategy's own managers.
        """
        match update:
            case BalanceUpdate():
                self._portfolio_manager.on_balance_update(update)
                self.on_balance(balance.account, balance)
            case PositionUpdate():
                self._portfolio_manager.on_position_update(update)
                self.on_position(position.account, position)
            case OrderUpdate():
                self._order_manager.on_order_update(update)
                self.on_order(order.account, order, type_)
            case TradeUpdate():
                self._order_manager.on_trade_update(update)
                self.on_trade(order.account, trade, type_)

    def _gather(self) -> None:
        if not self._is_gathered:
            # TODO: check if e.g. exchange balances and positions are ready, if backfilling is finished?
            # TODO: top strategy must have an account
            self.add_strategies()
            self.add_accounts()
            super()._gather()
        else:
            self.logger.debug(f"'{self.name}' has already gathered")

    def _set_engine(self, engine: TradeEngine):
        self._engine = engine

    # TODO: draft only
    # TODO: write order_price/order_quantity to trading store's df
    def place_orders(
        self,
        account: BaseAccount,
        product: BaseProduct,
        orders: list[BaseOrder] | BaseOrder,
    ):
        if not self.is_top_component():
            raise RuntimeError(f"Sub-strategy {self.name} cannot place orders")
        if not isinstance(orders, list):
            orders = [orders]
        if self.databoy.is_using_zmq():
            self.databoy._data_zmq.send(...)  # TODO: draft only
        else:
            venue = self._engine.get_venue(account.venue)
            venue._run_coroutine_threadsafe(venue.place_orders, args=(orders,))
        return orders

    # TODO
    def cancel_orders(
        self,
    ):
        pass

    # TODO
    def cancel_all_orders(self):
        pass

    # TODO
    def amend_orders(self):
        pass

    """
    ************************************************
    Override Methods
    Override these methods in your subclass to implement your custom behavior.
    ************************************************
    """

    def add_strategies(self):
        pass

    def add_accounts(self):
        pass

    def on_position(self, account, position):
        pass

    def on_balance(self, account, balance):
        pass

    def on_order(
        self, account, order, type_: Literal["submitted", "opened", "closed", "amended"]
    ):
        pass

    def on_trade(self, account, trade: dict, type_: Literal["partial", "filled"]):
        pass
