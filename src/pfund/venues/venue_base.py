from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Any,
    Generic,
    Literal,
    TypeVar,
    cast,
    TypeAlias,
)

if TYPE_CHECKING:
    from pfund_kit.utils.yaml import YAMLDocument
    from pfund.typing import (
        AccountName,
        ProductName,
        Currency,
        ProductKey,
        FullDataChannel,
    )
    from pfund.datas.resolution import Resolution
    from pfund.entities import (
        BaseAccount,
        BaseProduct,
        BaseBalance,
        BasePosition,
        BaseOrder,
        BaseMarket,
    )
    from pfund.venues.venue_config import VenueConfig
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.venues.adapter_base import BaseAdapter

import sys
import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Coroutine
from functools import cached_property

from pfund.typing import ProductKey
from pfund.entities.products.asset_type import AssetType
from pfund.venues.venue_metadata import VenueMetadata
from pfund.enums import Environment, TradingVenue, PrivateDataChannel


ConfigT = TypeVar("ConfigT", bound="VenueConfig")
MarketT = TypeVar("MarketT", bound="BaseMarket")
AccountT = TypeVar("AccountT", bound="BaseAccount")
BalanceT = TypeVar("BalanceT", bound="BaseBalance")
OrderT = TypeVar("OrderT", bound="BaseOrder")
ProductT = TypeVar("ProductT", bound="BaseProduct")
PositionT = TypeVar("PositionT", bound="BasePosition")
AnyVenue: TypeAlias = "BaseVenue[Any, Any, Any, Any, Any, Any, Any]"


class BaseVenue(
    ABC, Generic[ConfigT, MarketT, AccountT, BalanceT, OrderT, ProductT, PositionT]
):
    name: ClassVar[TradingVenue]
    adapter: ClassVar[BaseAdapter]
    Config: ClassVar[type[VenueConfig]]
    Market: ClassVar[type[BaseMarket]]
    Order: ClassVar[type[BaseOrder]]
    Account: ClassVar[type[BaseAccount]]
    Product: ClassVar[type[BaseProduct]]

    METADATA: ClassVar[VenueMetadata]
    MARKETS_FILENAME: ClassVar[str] = "markets.yml"

    def __init__(
        self,
        env: Literal[Environment.PAPER, Environment.LIVE, "PAPER", "LIVE"],
        config: ConfigT | None = None,
        settings: TradeEngineSettings | None = None,
    ):
        self._env = Environment[env.upper()]
        if self._env.is_simulated():
            raise ValueError(f"environment {self._env} is not supported")
        self._logger: logging.Logger = logging.getLogger(f"pfund.{self.name.lower()}")
        self._config: ConfigT = config or cast(ConfigT, self.Config())
        self._settings: TradeEngineSettings | None = settings
        self._accounts: dict[AccountName, AccountT] = {}
        self._products: dict[ProductName, ProductT] = {}

    def _run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Drive an async-native venue method from a synchronous context.

        Safe to call from sync code (scripts, REPL, notebooks). Raises RuntimeError
        if called from within a running event loop, since asyncio.run() cannot be
        nested — call the async variant (e.g. get_markets_async()) there instead.
        """
        method_name = sys._getframe(1).f_code.co_name
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # no running event loop -> safe to drive the coroutine with asyncio.run()
            return asyncio.run(coro)
        else:
            coro.close()  # avoid "coroutine was never awaited" RuntimeWarning
            raise RuntimeError(
                f"Cannot call {method_name}() from within a running event loop.\n"
                + f"Did you mean to call {method_name}_async()?"
            )

    @property
    def env(self) -> Literal[Environment.PAPER, Environment.LIVE]:
        return cast(Literal[Environment.PAPER, Environment.LIVE], self._env)

    # TODO
    def start(self):
        for channel in self._config.private_channels:
            self._add_private_channel(PrivateDataChannel[channel.lower()])

    # TODO
    def stop(self):
        pass

    @abstractmethod
    def get_markets(self, *args: Any, **kwargs: Any) -> dict[ProductKey, MarketT]:
        pass

    @cached_property
    def markets(self) -> dict[ProductKey, MarketT]:
        return (
            self._load_markets()
            if not self._config.refetch_markets
            else self._dump_markets()
        )

    @property
    def _markets_yml_file_path(self) -> Path:
        from pfund.config import get_config

        return get_config().data_path / self.MARKETS_FILENAME

    def _load_markets(self) -> dict[ProductKey, MarketT]:
        """Load markets.yml from disk."""
        from pfund_kit.utils.yaml import load

        file_path = self._markets_yml_file_path
        if not file_path.exists():
            return self._dump_markets()
        document: YAMLDocument = load(file_path) or {}
        markets: dict[ProductKey, MarketT] = {}
        for market_data in document.values():  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            market = self.Market(**market_data)
            key = ProductKey(
                symbol=market.symbol,
                asset_type=AssetType(market.asset_type),  # pyright: ignore[reportCallIssue]
            )
            markets[key] = market  # pyright: ignore[reportArgumentType]
        return markets

    def _dump_markets(self) -> dict[ProductKey, MarketT]:
        """Dump get_markets() result to markets.yml."""
        from pfund_kit.utils.yaml import dump

        markets: dict[ProductKey, MarketT] = self.get_markets()
        document = {
            f"{key.symbol}.{key.asset_type.value}": market.model_dump()
            for key, market in markets.items()
        }
        if document:
            dump(document, self._markets_yml_file_path)
        return markets

    def refresh_markets(self) -> dict[ProductKey, MarketT]:
        """Drop the cached markets and reload."""
        markets = self._dump_markets()
        self.__dict__["markets"] = (
            markets  # warm the cached_property  # pyright: ignore[reportIndexIssue]
        )
        return markets

    def _add_product(self, product: ProductT) -> None:
        if product.name not in self._products:
            self._products[product.name] = product
            self._logger.debug(
                f"added product name={product.name} symbol={product.symbol}"
            )
        else:
            raise ValueError(f"product name {product.name} is already registered")

    def _add_account(self, account: AccountT) -> None:
        if account.name not in self._accounts:
            self._accounts[account.name] = account
            self._logger.debug(f"added account name={account.name}")
        else:
            raise ValueError(f"account name {account.name} is already registered")

    @abstractmethod
    def add_channel(
        self,
        channel: FullDataChannel,
        *,
        channel_type: Literal["public", "private"] = "public",
    ) -> None: ...

    @abstractmethod
    def _create_market_data_channel(
        self, product: ProductT, resolution: Resolution
    ) -> FullDataChannel: ...

    @abstractmethod
    def _create_private_channel(
        self, channel: PrivateDataChannel
    ) -> FullDataChannel: ...

    def _add_market_data_channel(
        self, product: ProductT, resolution: Resolution
    ) -> None:
        full_channel: FullDataChannel = self._create_market_data_channel(
            product, resolution
        )
        self.add_channel(full_channel, channel_type="public")

    def _add_private_channel(self, channel: PrivateDataChannel) -> None:
        full_channel: FullDataChannel = self._create_private_channel(channel)
        self.add_channel(full_channel, channel_type="private")

    # def add_balance(
    #     self, venue: TradingVenue, acc: AccountName, ccy: Currency
    # ) -> BalanceT:
    #     venue = TradingVenue[venue.upper()]
    #     ccy = ccy.upper()
    #     if not (balance := self.get_balances(exch, acc=acc, ccy=ccy)):
    #         balance = CryptoBalance(ccy=ccy)
    #         self._logger.debug(f"added {balance}")
    #     return balance

    # def add_position(
    #     self, venue: TradingVenue, acc: str, pdt: ProductName
    # ) -> PositionT:
    #     venue = TradingVenue[venue.upper()]
    #     pdt = pdt.upper()
    #     if not (position := self.get_positions(exch, acc=acc, pdt=pdt)):
    #         account = self.get_account(exch, acc)
    #         product = self.add_product(exch, pdt=pdt)
    #         position = CryptoPosition(account, product)
    #         self._logger.debug(f"added {position}")
    #     return position

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
    ) -> ProductT:
        from pfeed.enums import DataSource
        from pfund.entities.products import ProductFactory, ProductBasis

        source = DataSource[cls.name.upper()]
        Product = cast("type[ProductT]", ProductFactory(source=source, basis=basis))
        return Product(
            source=source,
            basis=ProductBasis(basis=basis),
            exchange=exchange,
            name=name,
            symbol=symbol,
            specs=specs,
        )

    @property
    def account(self) -> AccountT | None:
        num_accounts = len(self._accounts)
        if num_accounts < 1:
            return None
        elif num_accounts > 1:
            raise ValueError(f"Expected exactly one account, got {num_accounts}")
        else:
            return list(self._accounts.values())[0]

    @property
    def accounts(self) -> dict[AccountName, AccountT]:
        return self._accounts

    @property
    def products(self) -> dict[ProductName, ProductT]:
        return self._products

    def get_account(self, name: AccountName) -> AccountT:
        return self.accounts[name]

    def get_product(self, name: ProductName) -> ProductT:
        return self.products[name]

    # TODO: api call
    def get_balances(self, name: AccountName) -> dict[Currency, BalanceT]: ...

    # TODO: api call
    def get_positions(self, name: AccountName) -> dict[ProductName, PositionT]: ...

    # TODO: api call
    def get_orders(self, name: AccountName) -> dict[ProductName, list[OrderT]]: ...

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
