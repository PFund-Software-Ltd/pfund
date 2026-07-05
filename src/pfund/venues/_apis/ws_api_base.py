from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    TypeAlias,
    Generic,
    TypeVar,
    cast,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pfund.entities import BaseProduct, BaseAccount
    from pfund.venues.adapter_base import BaseAdapter
    from pfund.datas.resolution import Resolution
    from pfund.venues.crypto_exchange import CryptoExchangeSigner
    from pfund.venues._apis.typing import ResponseData
    from pfund.typing import AccountName, FullDataChannel, ProductName

    WebSocketName: TypeAlias = str
    RawMessage: TypeAlias = dict[str, Any]
    Price: TypeAlias = float

import time
import queue
import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from msgspec import json
from websockets.asyncio.client import ClientConnection as WebSocket

from pfund.errors import WebSocketTimeoutError
from pfund.enums import Environment, TradingVenue, DataChannelType, PrivateDataChannel
from pfund.venues.venue_base import ConfigT, AccountT, ProductT


class NamedWebSocket(WebSocket):
    name: str  # pyright: ignore[reportUninitializedInstanceVariable]


class BaseWebSocketAPI(ABC, Generic[ConfigT, AccountT, ProductT]):
    venue: ClassVar[TradingVenue]
    _signer: ClassVar[CryptoExchangeSigner[Any]]

    VERSION: ClassVar[str | None] = None  # e.g. "v5" for str
    URLS: ClassVar[
        dict[Literal[Environment.PAPER, Environment.LIVE], dict[DataChannelType, str]]
    ] = {}
    PING_FREQ: ClassVar[int] = 20  # application-level ping to exchange (in seconds)
    RETRY_FREQ: ClassVar[int] = 3  # handshake retry backoff (in seconds)
    # handshake attempts per pass: on startup, giving up after this many aborts the
    # connection; on reconnect, it just ends the pass and a fresh pass keeps retrying.
    MAX_RETRIES: ClassVar[int] = 3

    def __init__(
        self,
        env: Literal[Environment.PAPER, Environment.LIVE, "PAPER", "LIVE"],
        config: ConfigT | None = None,
        read_only: bool = False,
    ):
        self._env = Environment[env.upper()]
        if self._env.is_simulated():
            raise ValueError(f"environment {self._env} is not supported")
        self._logger = logging.getLogger(f"pfund.{self.venue.lower()}")
        self._config: ConfigT = config or cast(ConfigT, self.venue.venue_class.Config())
        self._read_only = read_only

        self._callback: (
            Callable[[WebSocketName, RawMessage | ResponseData], Awaitable[None] | None]
            | None
        ) = None
        self._callback_raw_msg: bool = False

        self._products: dict[ProductName, ProductT] = {}
        self._accounts: dict[AccountName, AccountT] = {}
        self._channels: dict[DataChannelType, list[str]] = {
            DataChannelType.public: [],
            DataChannelType.private: [],
        }

        self._websockets: dict[WebSocketName, NamedWebSocket] = {}
        self._conn_tasks: set[asyncio.Task[None]] = set()
        self._num_conns = 0
        self._is_authenticated: dict[WebSocketName, bool] = defaultdict(bool)
        self._is_running = False
        self._last_ping_ts = time.time()

        self._queue: queue.Queue[Any] | None = None

    @property
    def env(self) -> Literal[Environment.PAPER, Environment.LIVE]:
        return cast(Literal[Environment.PAPER, Environment.LIVE], self._env)

    @property
    def adapter(self) -> BaseAdapter:
        return self.venue.venue_class.adapter

    def _set_queue(self, queue: queue.Queue[Any]) -> None:
        self._queue = queue

    @abstractmethod
    async def _subscribe(
        self,
        ws: NamedWebSocket,
        channels: list[FullDataChannel],
        channel_type: DataChannelType,
    ):
        pass

    @abstractmethod
    async def _unsubscribe(
        self,
        ws: NamedWebSocket,
        channels: list[FullDataChannel],
        channel_type: DataChannelType,
    ):
        pass

    @abstractmethod
    async def _authenticate(self, ws: NamedWebSocket, account: AccountT):
        pass

    @abstractmethod
    async def _ping(self):
        pass

    @abstractmethod
    async def _on_message(self, ws_name: str, raw_msg: bytes | str) -> Any:
        pass

    @abstractmethod
    def _create_market_data_channel(
        self, product: ProductT, resolution: Resolution
    ) -> FullDataChannel:
        pass

    def _create_private_channel(self, channel: PrivateDataChannel) -> FullDataChannel:
        channel = PrivateDataChannel[channel.lower()]
        return str(self.adapter(channel, group="channels"))

    @abstractmethod
    def _parse_message(
        self, ws_name: WebSocketName, msg: dict[str, Any]
    ) -> dict[str, Any]:
        pass

    @property
    def name(self) -> str:
        return self.venue

    @staticmethod
    def _convert_ms_to_seconds(ms: int | str) -> float:
        return int(ms) / 1000

    def set_callback(
        self,
        callback: Callable[
            [WebSocketName, RawMessage | ResponseData], Awaitable[None] | None
        ],
        raw_msg: bool = False,
    ):
        """
        Args:
            raw_msg:
                if True, the callback will receive the raw messages.
                if False, the callback will receive parsed messages.
        """
        self._callback = callback
        self._callback_raw_msg = raw_msg

    def _get_url(self, channel_type: DataChannelType) -> str:
        return self.URLS[self.env][DataChannelType[channel_type.lower()]]

    def add_account(self, account: AccountT) -> None:
        if account.env != self.env:
            raise ValueError(
                f"account env {account.env} does not match websocket env {self.env}"
            )
        if account in self._accounts.values():
            return
        if account.name not in self._accounts:
            self._accounts[account.name] = account
        else:
            raise ValueError(f"account name {account.name} has already been registered")

    def add_product(self, product: ProductT) -> None:
        if product in self._products.values():
            return
        if product.name not in self._products:
            self._products[product.name] = product
        else:
            raise ValueError(f"product name {product.name} has already been registered")

    def add_channel(
        self, channel: FullDataChannel, *, channel_type: Literal["public", "private"]
    ):
        channel_type: DataChannelType = DataChannelType[channel_type.lower()]
        if channel not in self._channels[channel_type]:
            self._channels[channel_type].append(channel)
            self._logger.debug(f"added {channel_type} channel '{channel}'")

    def _create_ws_name(self, account_name: str = ""):
        if not account_name:
            return "_".join([self.name, "ws"])
        else:
            return "_".join([account_name, "ws"])

    def _get_ws(self, ws_name: WebSocketName) -> NamedWebSocket | None:
        return self._websockets.get(ws_name, None)

    def _add_ws(self, ws_name: WebSocketName, ws: WebSocket) -> NamedWebSocket:
        if not ws_name.endswith("_ws"):
            raise ValueError(f'{ws_name=} must end with "_ws"')
        # HACK: assign ws_name to ws.name for conveniencec
        ws.name = ws_name  # pyright: ignore[reportAttributeAccessIssue]
        if ws_name in self._websockets:
            raise Exception(f"{ws_name} already exists")
        self._websockets[ws_name] = cast(NamedWebSocket, ws)
        return self._websockets[ws_name]

    async def _wait(
        self, target_condition: Callable[[], bool], description: str, timeout: int = 5
    ):
        while not target_condition():
            self._logger.debug(description)
            await asyncio.sleep(1)
            timeout -= 1
            if timeout <= 0:
                raise WebSocketTimeoutError(f"failed {description}")

    async def _checkup(self):
        # the connection task retries the handshake up to MAX_RETRIES times, sleeping
        # RETRY_FREQ between attempts, so the worst-case time to establish a socket is
        # MAX_RETRIES * RETRY_FREQ. readiness must wait at least that long, otherwise a
        # transient blip that recovers on a later retry would still abort startup.
        connect_timeout = self.MAX_RETRIES * self.RETRY_FREQ
        await self._wait(
            target_condition=lambda: (
                len(self._websockets) == self._num_conns and self._num_conns > 0
            ),
            description=f"waiting for all websockets to connect, ws_num={self._num_conns} websockets={list(self._websockets)}",
            timeout=connect_timeout,
        )
        if self._accounts:
            await self._wait(
                target_condition=lambda: all(
                    self._is_authenticated[
                        self._create_ws_name(account_name=account.name)
                    ]
                    for account in self._accounts.values()
                ),
                description=f"waiting for all accounts to be authenticated, is_authenticated={self._is_authenticated}",
            )

    def _cleanup(self):
        self._websockets.clear()
        self._conn_tasks.clear()
        self._num_conns = 0
        self._is_authenticated.clear()
        self._last_ping_ts = time.time()

    async def _run_background_tasks(self):
        while self._is_running:
            now = time.time()
            if now - self._last_ping_ts > self.PING_FREQ:
                await self._ping()
                self._last_ping_ts = now
            await asyncio.sleep(1)

    def _spawn_connect(self, url: str, account: AccountT | None = None) -> None:
        # NOTE: connections run as detached tasks (not awaited here) so that
        # _checkup() and _run_background_tasks() can run concurrently with the
        # live message loops in _connect_ws(). Joining the tasks (e.g. via a
        # TaskGroup) would block until every websocket closes.
        task = asyncio.create_task(self._connect_ws(url, account=account))
        self._conn_tasks.add(task)
        task.add_done_callback(self._conn_tasks.discard)

    async def connect(self):
        # error boundary, inherited by every venue (and venue facades like Bybit's
        # multi-category connect). Subclasses override _connect() with the work, never
        # this wrapper, so the try/except is written exactly once.
        try:
            await self._connect()
        except* asyncio.CancelledError:
            self._logger.warning(f"{self.name} connect() was cancelled")
            raise
        except* Exception:
            self._logger.exception(f"{self.name} connect() failed")

    async def _connect(self):
        for channel_type, channels in self._channels.items():
            if not channels:
                continue
            url = self._get_url(channel_type)
            if channel_type == DataChannelType.public:
                self._num_conns += 1
                self._spawn_connect(url)
            else:
                # one account uses one websocket connection (private streams are per-account)
                for account in self._accounts.values():
                    self._num_conns += 1
                    self._spawn_connect(url, account=account)
        if self._num_conns > 0:
            await self._checkup()
            self._is_running = True
            await self._run_background_tasks()
        else:
            self._logger.debug(
                f"{self.name} has no channels to subscribe to, nothing to connect"
            )

    async def _handshake(self, ws_name: WebSocketName, url: str) -> WebSocket | None:
        """Attempt the websocket handshake up to MAX_RETRIES times.

        Returns the live socket, or None if every attempt failed. The caller decides
        what None means: abort on startup, or keep retrying on reconnect.
        """
        from websockets.asyncio.client import connect

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._logger.debug(f"{ws_name} is connecting to {url}")
                return await connect(url)
            except Exception as err:
                self._logger.warning(
                    f"{ws_name} failed to connect ({err=}), retry {attempt}/{self.MAX_RETRIES}"
                )
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.RETRY_FREQ)
        return None

    async def _connect_ws(self, url: str, account: AccountT | None = None):
        from websockets.exceptions import (
            ConnectionClosed,
            ConnectionClosedError,
            ConnectionClosedOK,
        )

        ws_name = self._create_ws_name(account_name=account.name if account else "")
        if self._callback is None:
            self._logger.debug(f"websocket {ws_name} has no callback set")

        if account is None:
            channel_type = DataChannelType.public
        else:
            channel_type = DataChannelType.private

        # initial connection: bounded handshake retries. if it never comes up, give up so
        # _checkup() times out and startup aborts cleanly.
        ws = await self._handshake(ws_name, url)
        if ws is None:
            self._logger.error(
                f"{ws_name} could not connect after {self.MAX_RETRIES} retries, giving up"
            )
            return

        # connection loop: run the live socket's message loop until it drops, then keep
        # reconnecting until the venue recovers or disconnect() stops us.
        while True:
            self._logger.debug(f"{ws_name} is connected")
            try:
                async with ws:
                    ws = self._add_ws(ws_name, ws)

                    if channel_type == DataChannelType.private:
                        await self._authenticate(ws, account)  # pyright: ignore[reportArgumentType]

                    if channels := self._channels[channel_type]:
                        await self._subscribe(ws, channels, channel_type)

                    try:
                        async for msg in ws:
                            await self._on_message(ws_name, msg)
                    except ConnectionClosedOK:
                        self._logger.debug(f"{ws_name} closed normally")
                    except ConnectionClosedError as e:
                        self._logger.error(f"{ws_name} closed with error: {e}")
                    except ConnectionClosed as e:
                        self._logger.error(f"{ws_name} connection lost: {e}")
                    except Exception:
                        self._logger.exception(
                            f"{ws_name} unexpected error in message loop:"
                        )
            except Exception as err:
                # established but failed during auth/subscribe; fall through to reconnect.
                self._logger.warning(
                    f"{ws_name} error after connecting ({err=}), will reconnect"
                )
            finally:
                # drop the dead socket so the next pass can re-register a fresh one and
                # the registry keeps reflecting reality.
                self._websockets.pop(ws_name, None)
                self._is_authenticated[ws_name] = False

            # reconnect: retry forever until the socket comes back, unless disconnect()
            # has cleared _is_running (graceful shutdown), in which case we stop.
            ws = None
            while self._is_running and ws is None:
                self._logger.warning(f"{ws_name} reconnecting in {self.RETRY_FREQ}s")
                await asyncio.sleep(self.RETRY_FREQ)
                ws = await self._handshake(ws_name, url)
            if ws is None:
                self._logger.debug(f"{ws_name} stopped, not reconnecting")
                return

    async def disconnect(self, reason: str = ""):
        self._is_running = False
        for ws_name in list(self._websockets):
            if ws := self._get_ws(ws_name):
                await self._disconnect_ws(ws, reason=reason)
        # closing the sockets above ends the message loops; cancel any tasks still
        # pending (e.g. stuck mid-handshake) and wait for them to unwind.
        for task in list(self._conn_tasks):
            task.cancel()
        if self._conn_tasks:
            await asyncio.gather(*self._conn_tasks, return_exceptions=True)
        self._cleanup()

    async def _disconnect_ws(self, ws: NamedWebSocket, reason: str = ""):
        self._logger.warning(
            f"{ws.name} is disconnecting (state={ws.state.name}), {reason=}"
        )
        await ws.close(code=1000, reason=reason)
        await ws.wait_closed()
        self._websockets.pop(ws.name, None)
        self._is_authenticated[ws.name] = False
        self._logger.warning(f"{ws.name} is disconnected")

    async def _send(self, ws: NamedWebSocket, msg: dict[str, Any]):
        try:
            await ws.send(json.encode(msg), text=True)
            self._logger.debug(f"{ws.name} sent {msg}")
        except Exception:
            self._logger.exception(f"{ws.name} _send() exception:")

    def _emit_balance_update(
        self, ws_name: WebSocketName, response: ResponseData
    ) -> None:
        VenueClass = self.venue.venue_class
        ts, data = VenueClass._extract_ts_and_data_from_response_data(
            response, dtype=dict, ts_required=True
        )
        if data is None:
            return None
        account_name = ws_name.replace("_ws", "")
        if account_name not in self._accounts:
            self._logger.error(
                f"account {account_name} not found, existing accounts: {list(self._accounts.keys())}"
            )
            return None
        account = self._accounts[account_name]
        update = VenueClass._build_balance_update(
            ts, data, account.name, source="websocket"
        )
        if self._queue:
            self._queue.put(update)

    # def _validate_sequence_num(
    #     self,
    #     ws_name: str,
    #     pdt: str,
    #     seq_num: int,
    #     type_: Literal["quote", "position"] = "quote",
    # ) -> bool:
    #     if type_ == "quote":
    #         last_seq_num = self._last_quote_nums[pdt]
    #     else:
    #         raise NotImplementedError(f"sequence number {type_=} is not supported")

    #     if seq_num <= last_seq_num:
    #         self._logger.error(f"{pdt} {type_=} {seq_num=} <= {last_seq_num=}")
    #         self.disconnect(ws_name, reason=f"wrong {type_}_num")
    #         return False
    #     else:
    #         if type_ == "quote":
    #             self._last_quote_nums[pdt] = seq_num
    #         return True

    # def _process_position_msg(self, ws_name, msg, schema) -> dict:
    #     acc = ws_name
    #     positions = {"ts": None, "data": defaultdict(dict)}
    #     res = step_into(msg, schema["result"])
    #     res_type = type(res)
    #     if "ts" in schema:
    #         positions["ts"] = float(step_into(msg, schema["ts"])) * schema["ts_adj"]
    #     if res_type is list:
    #         for position in res:
    #             category = (
    #                 step_into(position, schema["category"])
    #                 if "category" in schema
    #                 else ""
    #             )
    #             epdt = step_into(position, schema["pdt"])
    #             pdt = self.adapter(epdt, group=category)
    #             qty = float(step_into(position, schema["data"]["qty"][0]))
    #             if qty == 0 and pdt not in self._products:
    #                 continue
    #             if "side" in schema:
    #                 eside = step_into(position, schema["side"])
    #                 side = self.adapter(eside, group="side")
    #             # e.g. BINANCE_USDT only returns position size (signed qty)
    #             elif "size" in schema:
    #                 side = sign(step_into(position, schema["size"]))
    #             positions["data"][pdt][side] = {}
    #             for k, (ek, *sequence) in schema["data"].items():
    #                 initial_value = self.adapter(step_into(position, ek))
    #                 v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
    #                 positions["data"][pdt][side][k] = v
    #     else:
    #         raise Exception(f"{self.exch} unhandled {res_type=}")
    #     zmq = self._get_zmq(ws_name)
    #     if zmq:
    #         zmq_msg = (3, 2, (self._bkr, self.exch, acc, positions))
    #         zmq.send(*zmq_msg)
    #     else:
    #         data = {
    #             "bkr": self._bkr,
    #             "exch": self.exch,
    #             "acc": acc,
    #             "channel": "position",
    #             "data": positions,
    #         }
    #         return data

    # def _process_balance_msg(self, ws_name, msg, schema):
    #     acc = ws_name
    #     balances = {"ts": None, "data": defaultdict(dict)}
    #     res = step_into(msg, schema["result"])
    #     res_type = type(res)
    #     if "ts" in schema:
    #         balances["ts"] = float(step_into(msg, schema["ts"])) * schema["ts_adj"]
    #     if res_type is list:
    #         for balance in res:
    #             eccy = step_into(balance, schema["ccy"])
    #             ccy = self.adapter(eccy)
    #             for k, (ek, *sequence) in schema["data"].items():
    #                 initial_value = self.adapter(step_into(balance, ek))
    #                 v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
    #                 balances["data"][ccy][k] = v
    #     else:
    #         raise Exception(f"{self.exch} unhandled {res_type=}")
    #     zmq = self._get_zmq(ws_name)
    #     if zmq:
    #         zmq_msg = (3, 1, (self._bkr, self.exch, acc, balances))
    #         zmq.send(*zmq_msg)
    #     else:
    #         data = {
    #             "bkr": self._bkr,
    #             "exch": self.exch,
    #             "acc": acc,
    #             "channel": "balance",
    #             "data": balances,
    #         }
    #         return data

    # def _process_order_msg(self, ws_name, msg, schema):
    #     acc = ws_name
    #     orders = {
    #         "ts": None,
    #         "data": defaultdict(list),
    #         "source": OrderUpdateSource.WSO,
    #     }
    #     res = step_into(msg, schema["result"])
    #     res_type = type(res)
    #     if "ts" in schema:
    #         orders["ts"] = float(step_into(msg, schema["ts"])) * schema["ts_adj"]
    #     if res_type is list:
    #         for order in res:
    #             category = (
    #                 step_into(order, schema["category"]) if "category" in schema else ""
    #             )
    #             epdt = step_into(order, schema["pdt"])
    #             pdt = self.adapter(epdt, group=category)
    #             update = {}
    #             for k, (ek, *sequence) in schema["data"].items():
    #                 group = k + "s" if k in ["tif", "side"] else ""
    #                 initial_value = self.adapter(step_into(order, ek), group=group)
    #                 v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
    #                 update[k] = v
    #             orders["data"][pdt].append(update)
    #     else:
    #         raise Exception(f"{self.exch} unhandled {res_type=}")
    #     zmq = self._get_zmq(ws_name)
    #     if zmq:
    #         zmq_msg = (2, 1, (self._bkr, self.exch, acc, orders))
    #         zmq.send(*zmq_msg)
    #     else:
    #         data = {
    #             "bkr": self._bkr,
    #             "exch": self.exch,
    #             "acc": acc,
    #             "channel": "order",
    #             "data": orders,
    #         }
    #         return data

    # def _process_trade_msg(self, ws_name, msg, schema):
    #     acc = ws_name
    #     trades = {
    #         "ts": None,
    #         "data": defaultdict(list),
    #         "source": OrderUpdateSource.WST,
    #     }
    #     res = step_into(msg, schema["result"])
    #     res_type = type(res)
    #     if "ts" in schema:
    #         trades["ts"] = float(step_into(msg, schema["ts"])) * schema["ts_adj"]
    #     if res_type is list:
    #         for trade in res:
    #             category = (
    #                 step_into(trade, schema["category"]) if "category" in schema else ""
    #             )
    #             epdt = step_into(trade, schema["pdt"])
    #             pdt = self.adapter(epdt, group=category)
    #             update = {}
    #             for k, (ek, *sequence) in schema["data"].items():
    #                 group = k + "s" if k in ["tif", "side"] else ""
    #                 initial_value = self.adapter(step_into(trade, ek), group=group)
    #                 v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
    #                 update[k] = v
    #             trades["data"][pdt].append(update)
    #     else:
    #         raise Exception(f"{self.exch} unhandled {res_type=}")
    #     zmq = self._get_zmq(ws_name)
    #     if zmq:
    #         zmq_msg = (2, 1, (self._bkr, self.exch, acc, trades))
    #         zmq.send(*zmq_msg)
    #     else:
    #         data = {
    #             "bkr": self._bkr,
    #             "exch": self.exch,
    #             "acc": acc,
    #             "channel": "trade",
    #             "data": trades,
    #         }
    #         return data
