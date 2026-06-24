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
    from pfund.venues.adapter_base import BaseAdapter
    from pfund.enums import AllAssetType

import logging
from abc import ABC, abstractmethod
from threading import Thread

from pfund.venues.venue_config import VenueConfig
from pfund.venues.venue_metadata import VenueMetadata
from pfund.enums import Environment, TradingVenue


class BaseVenue(ABC):
    name: ClassVar[TradingVenue]
    adapter: ClassVar[BaseAdapter]
    Order: ClassVar[type[BaseOrder]]
    Product: ClassVar[type[BaseProduct]]
    Account: ClassVar[type[BaseAccount]]
    METADATA: ClassVar[VenueMetadata]
    CONFIG_FILENAME: ClassVar[str] = "config.toml"
    MARKETS_FILENAME: ClassVar[str] = "markets.yml"

    def __init__(
        self, env: Environment | str, settings: TradeEngineSettings | None = None
    ):
        self._env = Environment[env.upper()]
        self._logger: logging.Logger = logging.getLogger(f"pfund.{self.name.lower()}")
        self._settings: TradeEngineSettings | None = settings
        self._accounts: dict[AccountName, BaseAccount] = {}
        self._products: dict[ProductName, BaseProduct] = {}
        self.config: VenueConfig = self._load_config()

    def _load_config(self) -> VenueConfig:
        pass

    # TODO: load markets.yml, also use it to set product tick_size, lot_size, fees etc.
    def _load_markets(self):
        pass

    def configure(self, persist: bool = False) -> VenueConfig:
        pass

    def start(self):
        pass

    def stop(self):
        pass

    # TODO
    # @abstractmethod
    # def add_product(self, *args: Any, **kwargs: Any) -> BaseProduct: ...

    # @abstractmethod
    # def add_account(self, *args: Any, **kwargs: Any) -> BaseAccount: ...

    def add_balance(
        self, veune: TradingVenue, acc: AccountName, ccy: Currency
    ) -> CryptoBalance:
        exch = CryptoExchange[veune.upper()]
        ccy = ccy.upper()
        if not (balance := self.get_balances(exch, acc=acc, ccy=ccy)):
            balance = CryptoBalance(ccy=ccy)
            self._logger.debug(f"added {balance}")
        return balance

    def add_position(
        self, exch: CryptoExchange, acc: str, pdt: ProductName
    ) -> CryptoPosition:
        exch, pdt = exch.upper(), pdt.upper()
        if not (position := self.get_positions(exch, acc=acc, pdt=pdt)):
            account = self.get_account(exch, acc)
            product = self.add_product(exch, pdt=pdt)
            position = CryptoPosition(account, product)
            self._logger.debug(f"added {position}")
        return position

    # TODO
    # def place_orders(self, orders: list[BaseOrder], threaded: bool=True, **kwargs: Any) -> list[BaseOrder]: ...
    # num_orders = 0
    # for o in orders:
    #     self._order_manager.on_submitted(o)
    #     num_orders += 1

    # if exchange.SUPPORT_PLACE_BATCH_ORDERS and num_orders > 1:
    #     place_orders = exchange.place_batch_orders
    #     orders = [orders]
    # else:
    #     place_orders = exchange.place_order

    # # REVIEW: performance issue if sending too many orders all at once and
    # # the exchange doesn't support batch orders
    # for order_s in orders:
    #     if not exchange.USE_WS_PLACE_ORDER:
    #         Thread(
    #             target=place_orders,
    #             args=(
    #                 account,
    #                 product,
    #                 order_s,
    #             ),
    #             daemon=True,
    #         ).start()
    #     # TODO
    #     else:
    #         ws_msg = place_orders(account, product, order_s)
    #         # NOTE: if exchange uses ws api to place order, it will return a ws_msg for ws
    #         # and this msg will be sent to the start_process() in connection_manager.py
    #         if ws_msg is not None:
    #             self._zmq.send(ws_msg)

    # TODO:
    # def cancel_orders(self, orders: list[BaseOrder], threaded: bool=True, **kwargs: Any)
    # num_orders = 0
    # for o in orders:
    #     self._order_manager.on_cancel(o)
    #     num_orders += 1

    # if exchange.SUPPORT_CANCEL_BATCH_ORDERS and num_orders > 1:
    #     cancel_orders = exchange.SUPPORT_CANCEL_BATCH_ORDERS
    #     orders = [orders]
    # else:
    #     cancel_orders = exchange.cancel_order

    # # REVIEW: performance issue if cancelling too many orders all at once and
    # # the exchange doesn't support batch orders
    # for order_s in orders:
    #     if not exchange.USE_WS_CANCEL_ORDER:
    #         Thread(
    #             target=cancel_orders,
    #             args=(
    #                 account,
    #                 product,
    #                 order_s,
    #             ),
    #             daemon=True,
    #         ).start()
    #     # TODO
    #     else:
    #         ws_msg = cancel_orders(account, product, order_s)
    #         # NOTE: if exchange uses ws api to cancel order, it will return a ws_msg for ws
    #         # and this msg will be sent to the start_process() in connection_manager.py
    #         if ws_msg is not None:
    #             self._zmq.send(ws_msg)

    # @abstractmethod
    # def amend_orders(self, *args: Any, threaded: bool=True, **kwargs: Any) -> list[BaseOrder]: ...

    # @abstractmethod
    # def cancel_all_orders(self, threaded: bool=True, reason: str | None=None): ...

    # TODO： assert account name to be unique
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

    def get_account(self, name: AccountName) -> BaseAccount:
        return self.accounts[name]

    def get_product(self, name: ProductName) -> BaseProduct:
        return self.products[name]

    # TODO: api call
    def get_balances(self, name: AccountName) -> dict[Currency, BaseBalance]: ...

    # TODO: api call
    def get_positions(self, name: AccountName) -> dict[ProductName, BasePosition]: ...

    # TODO: api call
    def get_orders(self, name: AccountName) -> dict[ProductName, list[BaseOrder]]: ...

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
    #     if self._settings:
    #         self._check_if_refetch_markets()
    #     for product in self._products.values():
    #         market_configs = self.load_market_configs()
    #         if product.symbol not in market_configs[product.category]:
    #             raise ValueError(
    #                 f"The symbol '{product.symbol}' is not found in the market configurations. "
    #                 + "It might be delisted, or your market configurations could be outdated. "
    #                 + "Please set 'refetch_markets=True' in TradeEngine's settings to refetch the latest market configurations."
    #             )
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
