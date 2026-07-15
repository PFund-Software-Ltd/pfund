from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from pfeed.storages.storage_config import StorageConfig

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
        Currency,
        ProductName,
    )

from abc import ABC, abstractmethod

from pfund.enums import Side
from pfund.utils.decorators import ray_method
from pfund.components.mixin import ComponentMixin
from pfund.components.strategies.strategy_meta import MetaStrategy
from pfund.managers import OrderManager, PortfolioManager, RiskManager


class BaseStrategy(ComponentMixin, ABC, metaclass=MetaStrategy):
    def __init__(self, *args: Any, **kwargs: Any):
        self._accounts: dict[AccountName, BaseAccount] = {}
        self.strategies: dict[str, BaseStrategy] = {}  # NOTE: sub-strategies
        self._is_substrategy = False
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
    def decide(self, X: IntoDataFrame) -> Literal[+1, -1, Side.BUY, Side.SELL, None]:
        pass

    @property
    def accounts(self) -> dict[AccountName, BaseAccount]:
        return self._accounts

    @property
    def balances(self) -> dict[AccountName, dict[Currency, BaseBalance]]:
        return self._portfolio_manager.balances

    @property
    def positions(self) -> dict[AccountName, dict[ProductName, BasePosition]]:
        return self._portfolio_manager.positions

    # TODO: add @property active_orders etc., get from order manager
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

    @ray_method
    def get_component(
        self, name: ComponentName
    ) -> Component | ActorProxy[Component] | None:
        return self.strategies.get(name, None) or super().get_component(name)

    @ray_method
    def get_components(self) -> list[Component | ActorProxy[Component]]:
        return [*self.strategies.values(), *super().get_components()]

    @ray_method
    def get_accounts(self) -> list[BaseAccount]:
        return list(self._accounts.values())

    def is_substrategy(self) -> bool:
        return self._is_substrategy

    @ray_method
    def _mark_as_substrategy(self):
        self._is_substrategy = True

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "strategies": list(self.strategies),
        }

    def _set_engine(self, engine: TradeEngine):
        self._engine = engine

    def add_account(self, account: BaseAccount) -> BaseAccount:
        if self.is_substrategy():
            raise ValueError(f"Sub-strategy '{self.name}' cannot add accounts")
        if account.env != self.env:
            raise ValueError(
                f"account env {account.env} does not match strategy env {self.env}"
            )
        if account.name not in self._accounts:
            self._accounts[account.name] = account
            self.logger.debug(f"added account name={account.name}")
        else:
            raise ValueError(f"account name {account.name} is already registered")
        return account

    def add_strategy(
        self,
        strategy: StrategyT | ActorProxy[StrategyT],
        resolution: str = "",
        name: str = "",
        storage_config: StorageConfig | None = None,
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> StrategyT | ActorProxy[StrategyT] | None:
        strategy._mark_as_substrategy()
        return self._add_component(
            component=strategy,
            resolution=resolution,
            name=name,
            df_form="long",
            storage_config=storage_config,
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )

    def signalize(self, X: IntoDataFrame) -> Signals:
        """Creates signals of this component

        Args:
            X: features df

        Returns:
            dict[ColumnName, Any]: The predicted signals.
        """
        signal = self.decide(X)
        if signal is not None:
            if signal not in [Side.BUY, Side.SELL]:
                raise ValueError(f"Invalid signal: {signal}")
            else:
                signal = Side(signal)
        signal_col = self._signal_cols[0]
        return {signal_col: signal}

    # TODO:
    def _on_update(self, update: Any):
        """Transport-agnostic sink for venue updates → this strategy's managers.

        Called with a fully-formed update object from BOTH paths: the engine's local
        in-process delivery, and databoy's _data_zmq recv (after reconstructing from the
        published dict via model_validate). Mirrors the engine's _run_updates_loop match,
        one level down into the strategy's own managers.
        """
        account_name = update.account
        account = self.accounts[account_name]
        match update:
            case BalanceUpdate():
                self._portfolio_manager.on_balance_update(update)
                self.on_balance(account, self.balances[account_name])
            case PositionUpdate():
                self._portfolio_manager.on_position_update(update)
                self.on_position(account, self.positions[account_name])
            case OrderUpdate():
                self._order_manager.on_order_update(update)
                self.on_order(order.account, order, type_)
            case TradeUpdate():
                self._order_manager.on_trade_update(update)
                self.on_trade(order.account, trade, type_)

    def _gather(self) -> None:
        if not self._is_gathered:
            self.add_accounts()
            self.add_strategies()
            if not self.is_substrategy() and not self._accounts:
                raise RuntimeError(f"{self.name} must have at least one account")
            super()._gather()
            self.set_signal_cols(["signal"])
        else:
            self.logger.debug(f"'{self.name}' has already gathered")

    # TODO: write order_price/order_quantity to trading store's df
    def place_orders(
        self,
        account: BaseAccount,
        product: BaseProduct,
        orders: list[BaseOrder] | BaseOrder,
    ):
        if self.is_substrategy():
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

    def on_position(self, account: BaseAccount, position: BasePosition):
        pass

    def on_balance(self, account: BaseAccount, balance: BaseBalance):
        pass

    def on_order(
        self,
        account: BaseAccount,
        order: BaseOrder,
        type_: Literal["submitted", "opened", "closed", "amended"],
    ):
        pass

    def on_trade(
        self, account: BaseAccount, trade: dict, type_: Literal["partial", "filled"]
    ):
        pass
