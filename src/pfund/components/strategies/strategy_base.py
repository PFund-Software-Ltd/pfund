from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload, override

if TYPE_CHECKING:
    import polars as pl
    from pfund_blocks.entities import (
        BaseAccount,
        BaseBalance,
        BasePosition,
        BaseProduct,
        BaseVenue,
    )
    from pfund_blocks.resolution import Resolution
    from pfund_blocks.typing import AccountName, Currency, ProductName, TradingVenue

    from pfund.components.actor_proxy import ActorProxy
    from pfund.engines.contexts.base_engine_context import BaseEngineContext
    from pfund.enums import RunMode
    from pfund.typing import (
        Component,
        ComponentName,
        ComponentProxy,
    )

from abc import ABC, abstractmethod

from pfund_blocks.entities.orders.order_base import BaseOrder

from pfund.components._mixins.component_mixin import ComponentMixin
from pfund.components.strategies.strategy_meta import MetaStrategy
from pfund.enums import ComponentType
from pfund.managers import OrderManager, PortfolioManager, RiskManager
from pfund.signals import Signals


class BaseStrategy(ComponentMixin, ABC, metaclass=MetaStrategy):
    def __init__(self, *args: Any, **kwargs: Any):
        self.__mixin_post_init__(ComponentType.strategy, *args, **kwargs)
        self._is_substrategy = False
        self._venues: dict[TradingVenue, BaseVenue] = {}
        self._accounts: dict[AccountName, BaseAccount] = {}
        self._order_manager = OrderManager()
        self._portfolio_manager = PortfolioManager()
        self._risk_manager = RiskManager()
        # NOTE: sub-strategies
        self.strategies: dict[str, BaseStrategy | ActorProxy[BaseStrategy]] = {}

    @abstractmethod
    def trade(self, X: pl.DataFrame) -> list[BaseOrder]:
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

    def get_component(self, name: ComponentName) -> Component | ComponentProxy:
        if name in self.strategies:
            return self.strategies[name]
        return super().get_component(name)

    def get_components(self) -> list[Component | ComponentProxy]:
        return [*self.strategies.values(), *super().get_components()]

    def get_accounts(self) -> list[BaseAccount]:
        return list(self._accounts.values())

    def get_venue(self, name: TradingVenue) -> BaseVenue | None:
        return self._venues.get(name, None)

    def is_substrategy(self) -> bool:
        return self._is_substrategy

    @override
    def _hydrate(
        self,
        name: ComponentName,
        run_mode: RunMode,
        resolution: Resolution | str,
        engine_context: BaseEngineContext[Any],
        df_form: Literal["wide", "long"],
        is_substrategy: bool = False,
    ):
        """
        Args:
            is_substrategy: True when added by a parent strategy, False when added by the engine.
                Passed through hydration so that an ActorProxy applies it only when its actor is created.
        """
        super()._hydrate(
            name=name,
            run_mode=run_mode,
            resolution=resolution,
            engine_context=engine_context,
            df_form=df_form,
        )
        self._is_substrategy = is_substrategy

    @override
    def to_dict(self) -> dict[str, Any]:
        """Adds strategy-only fields, kept out of _setup_fields() so they don't change _setup_sha256."""
        return {
            **super().to_dict(),
            "is_substrategy": self.is_substrategy(),
            # NOTE: name -> venue only, never the account object, it holds credentials
            "accounts": {
                name: account.venue for name, account in self._accounts.items()
            },
            "venues": sorted(self._venues),
        }

    def _setup_fields(self) -> dict[str, Any]:
        return {
            **super()._setup_fields(),
            # child stamps, not names, see ComponentMixin._setup_fields()
            "strategies": {
                name: strategy.get_artifact_stamp()
                for name, strategy in sorted(self.strategies.items())
            },
        }

    # TODO: venue should be read-only by default
    def add_venue(self):
        pass

    def add_account(self, account: BaseAccount) -> BaseAccount:
        if self.is_substrategy():
            raise ValueError(f"Sub-strategy '{self.name}' cannot add accounts")
        account._set_env(self.env)
        if account.name not in self._accounts:
            self._accounts[account.name] = account
            self.logger.debug(f"added account name={account.name}")
        else:
            raise ValueError(f"account name {account.name} is already registered")
        return account

    # remote strategy in -> remote strategy out, ray args are ignored (the proxy is already remote)
    @overload
    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: ActorProxy[StrategyT],
        resolution: str = "",
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ActorProxy[StrategyT]: ...

    # local strategy in + num_cpus -> wrapped into a remote strategy
    @overload
    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT,
        resolution: str = "",
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: float,
        **ray_kwargs: Any,
    ) -> ActorProxy[StrategyT]: ...

    # local strategy in, no num_cpus -> local strategy out
    @overload
    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT,
        resolution: str = "",
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: None = None,
        **ray_kwargs: Any,
    ) -> StrategyT: ...

    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT | ActorProxy[StrategyT],
        resolution: str = "",
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> StrategyT | ActorProxy[StrategyT]:
        """Add a child strategy to this strategy.

        Args:
            strategy: Strategy instance or remote strategy proxy to add.
            resolution: Resolution at which the child strategy runs. Inherits
                this strategy's resolution when omitted.
            name: Optional name for the child strategy.
            ray_actor_options: Options passed to the Ray actor.
            ray_kwargs: Ray actor constructor arguments. Providing these runs the
                child strategy remotely.

        Returns:
            The added child strategy or its remote proxy.
        """
        return self._add_component(
            component=strategy,
            resolution=resolution,
            name=name,
            df_form="long",
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )

    def to_signals(self, orders: list[BaseOrder]) -> Signals:
        """Converts the output of trade() to signals

        Args:
            orders: the orders returned by trade()

        Returns:
            Signals: The signals.
        """
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
        return Signals({signal_col: signal})

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
            venue = self.get_venue(account.venue)
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
