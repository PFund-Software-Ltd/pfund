from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Any

if TYPE_CHECKING:
    from pfund.typing import AccountName, ProductName, Currency
    from pfund.entities import (
        BaseAccount,
        BaseProduct,
        BaseBalance,
        BasePosition,
        BaseOrder,
    )
    from pfund.datas.resolution import Resolution
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.venues.adapter import Adapter

import logging
from abc import ABC, abstractmethod
from threading import Thread

from pfund.enums import Environment, TradingVenue


class BaseVenue(ABC):
    name: ClassVar[TradingVenue]
    adapter: ClassVar[Adapter]
    Order: ClassVar[type[BaseOrder]]
    Product: ClassVar[type[BaseProduct]]
    Account: ClassVar[type[BaseAccount]]

    def __init__(
        self, env: Environment | str, settings: TradeEngineSettings | None = None
    ):
        self._env = Environment[env.upper()]
        self._logger: logging.Logger = logging.getLogger("pfund")
        self._settings: TradeEngineSettings | None = settings
        self._accounts: dict[AccountName, BaseAccount] = {}
        self._products: dict[ProductName, BaseProduct] = {}

    # TODO
    # @abstractmethod
    # def add_product(self, *args: Any, **kwargs: Any) -> BaseProduct: ...

    # @abstractmethod
    # def add_account(self, *args: Any, **kwargs: Any) -> BaseAccount: ...

    # @abstractmethod
    # def place_orders(self, *args: Any, threaded: bool=True, **kwargs: Any) -> list[BaseOrder]: ...

    # @abstractmethod
    # def amend_orders(self, *args: Any, threaded: bool=True, **kwargs: Any) -> list[BaseOrder]: ...

    # @abstractmethod
    # def cancel_all_orders(self, threaded: bool=True, reason: str | None=None): ...

    # TODO
    # @classmethod
    # @abstractmethod
    # def create_account(cls, name: str = "", **kwargs: Any) -> BaseAccount: ...

    @classmethod
    def create_product(
        cls,
        basis: str,
        exchange: str = "",
        name: str = "",
        symbol: str = "",
        **specs: Any,
    ) -> BaseProduct:
        from pfeed.enums import DataSource
        from pfund.entities.products import ProductFactory

        source = DataSource[cls.name.upper()]
        Product = ProductFactory(source=source, basis=basis)
        return Product(
            basis=basis, exchange=exchange, name=name, symbol=symbol, specs=specs
        )

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def account(self) -> BaseAccount | None:
        num_accounts = len(self._accounts)
        if num_accounts < 1:
            return None
        elif num_accounts > 1:
            raise ValueError(f"Expected exactly one account, got {num_accounts}")
        else:
            return list(self._accounts.values())[0]

    @property
    def accounts(self) -> dict[AccountName, BaseAccount]:
        return self._accounts

    @property
    def products(self):
        return self._products

    # TODO
    @property
    def opened_orders(self): ...

    # TODO
    @property
    def submitted_orders(self): ...

    # TODO
    @property
    def closed_orders(self): ...

    # TODO
    @property
    def balances(self) -> dict[AccountName, dict[Currency, BaseBalance]]: ...

    # TODO
    @property
    def positions(self) -> dict[AccountName, dict[ProductName, BasePosition]]: ...

    def get_account(self, name: AccountName) -> BaseAccount:
        return self.accounts[name]

    def get_product(self, name: ProductName) -> BaseProduct:
        return self.products[name]

    # TODO
    def get_balances(
        self, name: AccountName, fetch: bool = False
    ) -> dict[Currency, BaseBalance]: ...

    # TODO
    def get_positions(
        self, name: AccountName, fetch: bool = False
    ) -> dict[ProductName, BasePosition]: ...

    # TODO
    def get_orders(
        self, name: AccountName, fetch: bool = False
    ) -> dict[ProductName, list[BaseOrder]]: ...

    # def _add_default_private_channels(self):
    #     for channel in PrivateDataChannel:
    #         self.add_channel(channel, channel_type="private")

    # def start(self):
    #     # TODO: check if all product names and account names are unique
    #     self._add_default_private_channels()
    # # for exch in self._accounts:
    #     for acc in self._accounts[exch]:
    #         balances = self.get_balances(exch, acc=acc, is_api_call=True)
    #         self._portfolio_manager.update_balances(exch, acc, balances)

    #         positions = self.get_positions(exch, acc=acc, is_api_call=True)
    #         self._portfolio_manager.update_positions(exch, acc, positions)

    #         orders = self.get_orders(exch, acc, is_api_call=True)
    #         self._order_manager.update_orders(exch, acc, orders)
    #     if self._settings.cancel_all_at["start"]:
    #         self.cancel_all_orders(reason="start")
    #     self._logger.debug(f"broker {self._name} started")

    # def stop(self):
    #     if self._settings.cancel_all_at["stop"]:
    #         self.cancel_all_orders(reason="stop")
    #     self._logger.debug(f"broker {self._name} stopped")

    # def _distribute_msgs(self, channel, topic, info):
    #     if ...:
    #         self.add_order(...)
    #         self._order_manager.update_orders(...)
    #     elif ...:
    #         self.add_position(...)
    #         self._portfolio_manager.update_positions(...)
    #     elif ...:
    #         self.add_balance(...)
    #         self._portfolio_manager.update_balances(...)
    #     # elif channel == 4:  # from api processes to connection manager
    #     #     self._connection_manager.handle_msgs(topic, info)
    #     #     if topic == 3 and self._settings.get('cancel_all_at', {}).get('disconnect', True):  # on disconnected
    #     #         self.cancel_all_orders(reason='disconnect')

    # def schedule_jobs(self, scheduler: BackgroundScheduler):
    #     scheduler.add_job(self.reconcile_balances, "interval", seconds=10)
    #     scheduler.add_job(self.reconcile_positions, "interval", seconds=10)
    #     scheduler.add_job(self.reconcile_orders, "interval", seconds=10)
    #     scheduler.add_job(self.reconcile_trades, "interval", seconds=10)
    #     for manager in [self._order_manager, self._portfolio_manager]:
    #         manager.schedule_jobs(scheduler)
