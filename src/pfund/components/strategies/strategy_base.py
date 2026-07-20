from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from pfeed.storages.storage_config import StorageConfig

    from pfund.components.actor_proxy import ActorProxy
    from pfund.entities.accounts.account_base import BaseAccount
    from pfund.entities.balances.balance_base import BaseBalance
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

from pfund.entities.orders.order_base import BaseOrder
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
    def trade(self, X: IntoDataFrame) -> list[BaseOrder]:
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

    def _identity_fields(self) -> dict[str, Any]:
        return {
            **super()._identity_fields(),
            "strategies": sorted(self.strategies),
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
        """Add a child strategy to this strategy.

        Args:
            strategy: Strategy instance or remote strategy proxy to add.
            resolution: Resolution at which the child strategy runs. Inherits
                this strategy's resolution when omitted.
            name: Optional name for the child strategy.
            storage_config: Per-strategy storage configuration. Inherits this
                strategy's storage configuration when omitted.
            ray_actor_options: Options passed to the Ray actor.
            ray_kwargs: Ray actor constructor arguments. Providing these runs the
                child strategy remotely.

        Returns:
            The added child strategy or its remote proxy. Returns ``None`` when
            adding a local child strategy to a remote parent strategy.
        """
        strategy._mark_as_substrategy()
        strategy = self._add_component(
            component=strategy,
            resolution=resolution,
            name=name,
            df_form="long",
            storage_config=storage_config,
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )
        # NOTE: must be called AFTER hydration, because strategy may be an ActorProxy,
        # need to mark it as a substrategy after its creation
        if strategy is not None:
            strategy._mark_as_substrategy()
        return strategy

    def signalize(self, X: IntoDataFrame) -> Signals:
        """Creates signals of this component

        Args:
            X: features df

        Returns:
            dict[ColumnName, Any]: The predicted signals.
        """
        orders = self.trade(X)
        if orders is None:
            orders = []
        if not isinstance(orders, (list, set)):
            raise TypeError(
                f"Expected list or set returned from trade(), got {type(orders)}"
            )
        if not orders:
            signal = None
        else:
            if not isinstance(orders[0], BaseOrder):
                raise TypeError(
                    f"Expected list of Order objects, got {type(orders[0])}"
                )
            sides = {o.side for o in orders}
            if len(sides) != 1:
                raise ValueError(f"All orders should have the same side, got: {sides}")
            signal = sides.pop()
        signal_col = self._signal_cols[0]
        # TODO: add 'order_type', 'order_price', 'order_quantity' to signals
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

    @override
    def _get_default_signal_cols(self, num_cols: int) -> list[str]:
        if num_cols != 1:
            raise ValueError(
                f"{self.name} strategy must have exactly one signal column"
            )
        return [self.name] if self.is_substrategy() else ["signal"]

    def _gather(self) -> None:
        if not self._is_gathered:
            self.add_accounts()
            self.add_strategies()
            if not self.is_substrategy() and not self._accounts:
                raise RuntimeError(f"{self.name} must have at least one account")
            super()._gather()
            self.set_signal_cols(self._get_default_signal_cols(num_cols=1))
        else:
            self.logger.debug(f"'{self.name}' has already gathered")

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
        if self._databoy.is_using_zmq():
            self._databoy._data_zmq.send(...)  # TODO: draft only
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
