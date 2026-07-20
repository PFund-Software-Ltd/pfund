# pyright: reportUnusedParameter=false
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Callable,
    Any,
    Generic,
    Literal,
    TypeVar,
    cast,
    TypeAlias,
)

if TYPE_CHECKING:
    from pfund.typing import (
        AccountName,
        ProductName,
        Currency,
        FullDataChannel,
    )
    from pfund.datas.data_market import MarketData
    from pfund.entities import (
        BaseAccount,
        BaseProduct,
        BaseBalance,
        BasePosition,
        BaseOrder,
        BaseMarket,
    )
    from pfund.venues.venue_config import VenueConfig
    from pfund.venues.adapter_base import BaseAdapter
    from pfund.venues._apis.typing import Result, ResponseData

import sys
import queue
import asyncio
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from threading import Thread
from collections.abc import Coroutine
from concurrent.futures import Future

from pfund.entities.balances.balance_base import BalanceUpdate, BalanceUpdateSource
from pfund.errors import NotSupportedByVenueError
from pfund.entities.products.product_base import ProductKey
from pfund.venues.venue_metadata import VenueMetadata
from pfund.enums import Environment, TradingVenue


ConfigT = TypeVar("ConfigT", bound="VenueConfig")
MarketT = TypeVar("MarketT", bound="BaseMarket")
AccountT = TypeVar("AccountT", bound="BaseAccount")
BalanceT = TypeVar("BalanceT", bound="BaseBalance")
BalanceSnapshotT = TypeVar("BalanceSnapshotT", bound="BaseBalance.Snapshot")
OrderT = TypeVar("OrderT", bound="BaseOrder")
ProductT = TypeVar("ProductT", bound="BaseProduct")
PositionT = TypeVar("PositionT", bound="BasePosition")
PositionSnapshotT = TypeVar("PositionSnapshotT", bound="BasePosition.Snapshot")
DataT = TypeVar("DataT", list[dict[str, Any]], dict[str, Any])
AnyVenue: TypeAlias = "BaseVenue[Any, Any, Any, Any, Any, Any, Any, Any, Any]"


class BaseVenue(
    ABC,
    Generic[
        ConfigT,
        MarketT,
        AccountT,
        ProductT,
        OrderT,
        BalanceT,
        BalanceSnapshotT,
        PositionT,
        PositionSnapshotT,
    ],
):
    name: ClassVar[TradingVenue]
    adapter: ClassVar[BaseAdapter]
    Config: ClassVar[type[VenueConfig]]
    Market: ClassVar[type[BaseMarket]]
    Account: ClassVar[type[BaseAccount]]
    Balance: ClassVar[type[BaseBalance]]
    Order: ClassVar[type[BaseOrder]]
    Product: ClassVar[type[BaseProduct]]
    Position: ClassVar[type[BasePosition]]

    METADATA: ClassVar[VenueMetadata]
    MARKETS_FILENAME: ClassVar[str] = "markets.yml"

    def __init__(
        self,
        env: Literal[
            Environment.PAPER,
            Environment.LIVE,
            Environment.SANDBOX,
            "PAPER",
            "LIVE",
            "SANDBOX",
        ],
        config: ConfigT | None = None,
        read_only: bool = False,
    ):
        self._env = Environment[env.upper()]
        self._logger: logging.Logger = logging.getLogger(f"pfund.{self.name.lower()}")
        self._config: ConfigT = config or cast(ConfigT, self.Config())
        self._read_only = read_only
        self._accounts: dict[AccountName, AccountT] = {}
        self._products: dict[ProductName, ProductT] = {}
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread = Thread(
            target=self._run_loop, name=f"{self.name}_loop", daemon=True
        )
        self._queue: queue.Queue[Any] | None = None
        if self.METADATA.has_markets:
            markets_file = self._create_markets_yml_file_path()
            if self._config.refetch_markets or not markets_file.exists():
                self.refetch_markets()

    @property
    def env(self) -> Literal[Environment.PAPER, Environment.LIVE, Environment.SANDBOX]:
        return cast(
            Literal[Environment.PAPER, Environment.LIVE, Environment.SANDBOX], self._env
        )

    @classmethod
    def _create_markets_yml_file_path(cls, data_path: Path | None = None) -> Path:
        if data_path is None:
            from pfund.config import get_config

            data_path = get_config().data_path
        return data_path / cls.name / cls.MARKETS_FILENAME

    def _set_queue(self, queue: queue.Queue[Any]) -> None:
        """Set the queue for the venue to send data back to the trade engine."""
        self._queue = queue

    def _log_api_error(
        self,
        error: str,
        account: BaseAccount | None = None,
        method: str | None = None,
    ) -> None:
        """Log a failed venue API call, auto-detecting the calling method's name.

        Shared by get_balances/get_positions/etc. so the failure log line is
        standardized across endpoints. ``account`` is optional since some
        endpoints (e.g. public market data) aren't tied to an account. ``method``
        lets an internal helper pass the real caller's name (its own caller),
        since ``sys._getframe(1)`` would otherwise resolve to the helper.
        """
        method_name = method or sys._getframe(1).f_code.co_name
        suffix = f" for account={account.name}" if account is not None else ""
        self._logger.error(f"{method_name} failed{suffix}: {error}")

    def _extract_ts_and_data_from_api_result(
        self,
        result: Result,
        dtype: type[DataT],
        account: BaseAccount | None = None,
        ts_required: bool = True,
    ) -> tuple[float | None, DataT | None]:
        """Validate a REST `Result` and return its `ts` and `data` payload as `dtype`.

        Strips the `Result` success/error envelope — returning None (logging a
        standardized error) when the call failed, a transient operational
        condition callers can early-return on — then delegates the `ts`/`data`
        extraction to `_extract_ts_and_data_from_response_data`.
        `data = self._extract_ts_and_data_from_api_result(result, dtype=list)`
        is statically typed as `list | None`.
        """
        method_name = sys._getframe(1).f_code.co_name
        if not result["success"]:
            self._log_api_error(result["error"], account=account, method=method_name)
            return None, None
        return self._extract_ts_and_data_from_response_data(
            result["response"],
            dtype,
            ts_required=ts_required,
            _method_name=method_name,
        )

    @staticmethod
    def _extract_ts_and_data_from_response_data(
        response_data: ResponseData,
        dtype: type[DataT],
        ts_required: bool = True,
        _method_name: str | None = None,
    ) -> tuple[float | None, DataT | None]:
        """Validate a parsed `ResponseData` and return its `ts` and `data` as `dtype`.

        Shared core: the REST path reaches here through
        `_extract_ts_and_data_from_api_result` after the `Result` envelope is
        stripped; ws handlers, which already hold a parsed `ResponseData`, call
        this directly. Raises `TypeError` when `data` isn't the expected
        `list`/`dict` shape, since that means our parsing contract is broken (a
        bug), not a venue hiccup.
        """
        method_name = _method_name or sys._getframe(1).f_code.co_name
        ts = response_data.get("ts")
        data = response_data["data"]
        if not isinstance(data, dtype):
            raise TypeError(
                f"{method_name} expected data to be a {dtype.__name__} "
                + f"but got {type(data).__name__}"
            )
        if ts_required and ts is None:
            raise ValueError(f"{method_name} expected ts but got None")
        return ts, data

    async def get_markets(self, *args: Any, **kwargs: Any) -> dict[ProductKey, MarketT]:
        raise NotSupportedByVenueError(f"{self.name} does not support get_markets")

    def get_markets_sync(self, *args: Any, **kwargs: Any) -> dict[ProductKey, MarketT]:
        return self._run_async(self.get_markets(*args, **kwargs))

    def refetch_markets(self) -> None:
        """Dump get_markets() result to markets.yml."""
        from pfund_kit.utils.yaml import dump

        markets: dict[ProductKey, MarketT] = self.get_markets_sync()
        file_path = self._create_markets_yml_file_path()
        if not markets:
            self._logger.warning(
                f"{self.name} get_markets() returned no markets; "
                + f"keeping existing {file_path} (may be stale)"
            )
            return
        document = {str(key): market.model_dump() for key, market in markets.items()}
        dump(document, file_path)

    @abstractmethod
    def add_channel(
        self,
        channel: FullDataChannel,
        *,
        channel_type: Literal["public", "private"] = "public",
    ) -> None: ...

    @abstractmethod
    def _add_market_data_channel(self, data: MarketData) -> None: ...

    @abstractmethod
    def _add_private_channels(self) -> None: ...

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
    def account(self) -> AccountT:
        num_accounts = len(self._accounts)
        if num_accounts != 1:
            raise ValueError(f"Expected exactly one account, got {num_accounts}")
        else:
            return list(self._accounts.values())[0]

    @property
    def accounts(self) -> dict[AccountName, AccountT]:
        return self._accounts

    def get_account(self, name: AccountName) -> AccountT:
        return self.accounts[name]

    def add_account(self, account: AccountT) -> None:
        if account.env != self._env:
            raise ValueError(
                f"account env {account.env} does not match venue env {self._env}"
            )
        if account in self._accounts.values():
            return
        if account.name not in self._accounts:
            self._accounts[account.name] = account
            self._logger.debug(f"added account name={account.name}")
        else:
            raise ValueError(f"account name {account.name} is already registered")

    @property
    def products(self) -> dict[ProductName, ProductT]:
        return self._products

    def get_product(self, name: ProductName) -> ProductT:
        return self.products[name]

    def add_product(self, product: ProductT) -> None:
        if product in self._products.values():
            return
        if product.name not in self._products:
            self._products[product.name] = product
            self.adapter.add_mapping(
                group="products",
                internal=product.name,
                external=product.symbol,
            )
            self._logger.debug(f"added {product.desc_str()}")
            if self.METADATA.has_markets and product.market is None:
                file_path = self._create_markets_yml_file_path()
                product.load_market(file_path)
                if product.market is None:
                    self._logger.error(
                        f"no market found for {product.desc_str()} "
                        + f"in {file_path}; the product might be delisted, or the file is outdated "
                        + "- set refetch_markets=True in the venue config to refetch"
                    )
        else:
            raise ValueError(f"product name {product.name} is already registered")

    @abstractmethod
    async def _get_balances(self, account: AccountT) -> Result:
        """Venue-specific raw balance fetch (e.g. a signed REST call).

        Returns a parsed ``Result`` whose ``response['data']`` is a dict with:
          - ``account``: the account-level consolidated balance (a snapshot dict)
          - ``balances``: a list of per-currency snapshot dicts (each carrying a
            ``currency`` key)
        All the shared parsing/snapshotting/dispatch lives in ``get_balances``.
        """
        ...

    async def get_balances(
        self, account: AccountT
    ) -> BalanceUpdate[BalanceSnapshotT] | None:
        """Fetch the account's balances: per-currency snapshots + the account total.

        A per-currency "balance" is a holding of a unit of account (USD, USDT, BTC
        as a wallet coin, ...) — the money you settle in, keyed by ``Currency``;
        ``account_balance`` is the account-level consolidated total. Neither is
        directional exposure to a product: instrument holdings like AAPL or a BTC
        perp live under ``get_positions``, keyed by ``Product``.
        """
        result: Result = await self._get_balances(account)
        ts, data = self._extract_ts_and_data_from_api_result(
            result, dtype=dict, account=account, ts_required=True
        )
        if data is None:
            return None
        update = self._build_balance_update(
            ts, data, account.name, source="get_balances"
        )
        if self._queue:
            self._queue.put(update)
        return update

    def get_balances_sync(
        self, account: AccountT
    ) -> BalanceUpdate[BalanceSnapshotT] | None:
        return self._run_async(self.get_balances(account))

    @classmethod
    def _build_balance_update(
        cls,
        ts: float | None,
        data: dict[str, Any],
        account_name: AccountName,
        source: BalanceUpdateSource,
    ) -> BalanceUpdate[BalanceSnapshotT]:
        """Assemble a `BalanceUpdate` from an already-extracted `{account, balances}`.

        Shared by the REST `get_balances` and the ws wallet-push path: each
        currency snapshot is keyed by its ``currency`` (popped from the rest of
        the snapshot fields), and the optional account-level total is snapshotted
        separately. Classmethod (uses ``cls.Balance.Snapshot``, no instance
        state) so the ws api can reach it via ``self.venue.venue_class``. Builds
        only — the caller is responsible for putting the update on its queue,
        since REST and ws hold their queues on different objects.
        """
        snapshots: dict[Currency, BalanceSnapshotT] = {}
        account_balance: BalanceSnapshotT | None = None
        for balance in data["balances"]:
            currency = balance["currency"]
            snapshots[currency] = cast(
                "BalanceSnapshotT",
                cls.Balance.Snapshot(
                    updated_at=ts,
                    **{k: v for k, v in balance.items() if k != "currency"},
                ),
            )
        if "account" in data:
            account_balance = cast(
                "BalanceSnapshotT",
                cls.Balance.Snapshot(updated_at=ts, **data["account"]),
            )
        return BalanceUpdate[BalanceSnapshotT](
            ts=cast(float, ts),
            account=account_name,
            snapshots=snapshots,
            account_balance=account_balance,
            source=source,
        )

    # TODO:
    async def get_positions(
        self, account: AccountT
    ) -> dict[ProductName, PositionT]: ...

    def get_positions_sync(self, account: AccountT) -> dict[ProductName, PositionT]:
        return self._run_async(self.get_positions(account))

    # TODO:
    async def get_orders(
        self, account: AccountT
    ) -> dict[ProductName, list[OrderT]]: ...

    def get_orders_sync(self, account: AccountT) -> dict[ProductName, list[OrderT]]:
        return self._run_async(self.get_orders(account))

    def _run_coroutine_threadsafe(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Future[Any] | None:
        """Schedule `func(*args, **kwargs)` on the venue's event loop from another thread.

        Args:
            timeout:
                None (default): fire-and-forget — returns the Future immediately; a
                    coroutine failure is logged by the done-callback.
                float: block up to `wait` seconds for completion, then return None. A
                    timeout is logged here (the done-callback hasn't fired yet); a
                    coroutine failure is already logged by the done-callback.
        """

        def _callback(future: Future[Any]) -> None:
            func_name = func.__name__
            if future.cancelled():
                self._logger.warning(f"{self.name} {func_name}() was cancelled")
                return
            exc = future.exception()
            if exc is not None:
                self._logger.error(f"{self.name} {func_name}() failed", exc_info=exc)

        kwargs = kwargs or {}
        future: Future[Any] = asyncio.run_coroutine_threadsafe(
            func(*args, **kwargs), self._loop
        )
        future.add_done_callback(_callback)
        if timeout is None:
            return future
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            self._logger.warning(
                f"{self.name} {func.__name__}() did not finish within {timeout}s"
            )
        except Exception:
            pass  # coroutine failure already logged by the done-callback
        return None

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

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()  # NOTE: blocking

    @abstractmethod
    def connect(self):
        """Connect to the venue's API (e.g. websocket)"""
        ...

    @abstractmethod
    def disconnect(self, reason: str = ""):
        """Disconnect from the venue's API (e.g. websocket)"""
        ...

    def start(self):
        if self._queue is None:
            raise ValueError("Queue not set")
        self._loop_thread.start()
        if self._accounts:
            self._add_private_channels()
        self.connect()
        # TODO
        # if 'start' in self._config.cancel_all_at:
        #     self.cancel_all_orders(reason="start")

    def stop(self):
        if self._loop_thread.is_alive():
            self.disconnect(reason="venue stopped")
        # TODO
        # if 'stop' in self._config.cancel_all_at:
        #     self.cancel_all_orders(reason="stop")
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

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
