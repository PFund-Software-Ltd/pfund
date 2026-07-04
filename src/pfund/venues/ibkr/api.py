# ruff: noqa: E731
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, ClassVar, cast, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from ibapi.contract import Contract, ContractDetails

    from pfund.venues._apis.typing import ResponseData
    from pfund.venues.adapter_base import BaseAdapter
    from pfund.venues.ibkr.account import InteractiveBrokersAccount
    from pfund.venues.ibkr.product import InteractiveBrokersProduct
    from pfund.typing import AccountName, FullDataChannel, ProductName

import queue
import logging
import asyncio
from pprint import pformat
from collections import defaultdict
from threading import Thread

from pfund.errors import InteractiveBrokersError
from pfund.datas.resolution import Resolution
from pfund.datas.timeframe import Timeframe
from pfund.venues.ibkr.config import InteractiveBrokersConfig
from pfund.venues.ibkr._ibapi.client import InteractiveBrokersClient as IBClient
from pfund.venues.ibkr._ibapi.wrapper import InteractiveBrokersWrapper as IBWrapper
from pfund.venues.ibkr._ibapi.wrapper import *  # noqa: F403  # pyright: ignore[reportWildcardImportFromLibrary]
from pfund.enums import (
    Environment,
    TradingVenue,
    DataChannel,
    DataChannelType,
    PrivateDataChannel,
    TraditionalAssetType,
)


class InteractiveBrokersAPI(IBClient, IBWrapper):  # pyright: ignore[reportUnsafeMultipleInheritance]
    venue: ClassVar[TradingVenue] = TradingVenue.IBKR

    # default channels to subscribe to other than the ones in PrivateDataChannel
    DEFAULT_PRIVATE_CHANNELS: ClassVar[Sequence[str]] = ("reqAccountSummary",)

    ASSET_TYPES_WITHOUT_TICK_BY_TICK_DATA = [
        TraditionalAssetType.OPT,
        # EXTEND
        # TraditionalAssetType.INDEX,
        # TraditionalAssetType.COMBO
    ]
    # asset types that cannot subscribe to 'Last'/'AllLast' in reqTickByTickData()
    ASSET_TYPES_WITHOUT_TICK_BY_TICK_LAST_DATA = [
        TraditionalAssetType.OPT,
        TraditionalAssetType.FX,
    ]

    def __init__(
        self,
        env: Literal[Environment.PAPER, Environment.LIVE, "PAPER", "LIVE"],
        config: InteractiveBrokersConfig | None = None,
        timeout: float = 5.0,
    ):
        """
        Args:
            timeout:
                Seconds a REST-like request (see _request) waits for its complete
                response — the callback rows terminated by a *End callback — before
                raising TimeoutError. Applied per request, from the moment the
                request is sent. It does NOT apply to subscriptions/streams, and a
                timed-out request is NOT retried.
        """
        IBClient.__init__(self)
        IBWrapper.__init__(self)
        self._env = Environment[env.upper()]
        if self._env.is_simulated():
            raise ValueError(f"environment {self._env} is not supported")
        self._logger = logging.getLogger(f"pfund.{self.venue.lower()}")
        self._config = config or InteractiveBrokersConfig()
        self._timeout = timeout

        self._callback: Callable[[str], Awaitable[None] | None] | None = None
        self._callback_raw_msg: bool = False

        self._products: dict[ProductName, InteractiveBrokersProduct] = {}
        self._accounts: dict[AccountName, InteractiveBrokersAccount] = {}
        self._channels: dict[DataChannelType, list[str]] = {
            DataChannelType.public: [],
            DataChannelType.private: [],
        }

        self._queue: queue.Queue[Any] | None = None

        self._ib_thread = Thread(name="IBClientThread", target=self.run, daemon=True)
        # TODO: clean up, outdated from previous version
        self._ib_params_for_channels_subscription = {}

        # since reqMktData will subscribe to bid/ask + last price/quantity automatically,
        # use this to save down which tick types the system has subscribed
        self._subscribed_market_data_tick_types = defaultdict(list)

    @property
    def env(self) -> Literal[Environment.PAPER, Environment.LIVE]:
        return cast(Literal[Environment.PAPER, Environment.LIVE], self._env)

    @property
    def name(self) -> str:
        return self.venue

    @property
    def adapter(self) -> BaseAdapter:
        return self.venue.venue_class.adapter

    """
    **********************************************************
    REST-based like methods
    **********************************************************
    """

    async def _request(
        self,
        request_id: int,
        func: Callable[..., Any],
    ) -> list[Any]:
        if request_id in self._pending_responses:
            raise ValueError(f"{request_id=} is already pending")
        future: asyncio.Future[list[Any]] = asyncio.get_running_loop().create_future()
        self._pending_responses[request_id] = future
        try:
            func()
            return await asyncio.wait_for(future, self._timeout)
        finally:
            self._pending_responses.pop(request_id, None)
            self._partial_responses.pop(request_id, None)

    def _on_response(self, request_id: int, data: Any) -> None:
        """Accumulate one callback row for a pending request"""
        if request_id in self._pending_responses:
            self._partial_responses[request_id].append(data)

    def _on_response_end(self, request_id: int) -> None:
        """Resolve a pending request with its accumulated rows"""
        if (future := self._pending_responses.get(request_id)) is None:
            return

        data: list[Any] = self._partial_responses.pop(request_id, [])

        def _resolve() -> None:
            if not future.done():
                future.set_result(data)

        future.get_loop().call_soon_threadsafe(_resolve)

    def _on_response_error(self, request_id: int, error: str) -> None:
        """Fail a pending request so its awaiter raises instead of waiting for the timeout"""
        if (future := self._pending_responses.get(request_id)) is None:
            return

        def _fail() -> None:
            if not future.done():
                future.set_exception(InteractiveBrokersError(error))

        future.get_loop().call_soon_threadsafe(_fail)

    # TODO: parse the raw data (list[ContractDetails]) and return Result just like rest api
    # and handle the raised InteractiveBrokersError
    async def get_contract_details(
        self, contract: Contract, request_id: int | None = None
    ) -> list[ContractDetails]:
        request_id = request_id or self._next_request_id()
        func = lambda: self.reqContractDetails(request_id, contract)
        return await self._request(request_id, func)

    """
    **********************************************************
    WebSocket-based like methods
    **********************************************************
    """

    def _set_queue(self, queue: queue.Queue[Any]) -> None:
        self._queue = queue

    def set_callback(
        self,
        callback: Callable[[str], Awaitable[None] | None],
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

    def add_account(self, account: InteractiveBrokersAccount):
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

    def add_product(self, product: InteractiveBrokersProduct):
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
            self._logger.debug(f"added {channel_type} channel {channel}")

    def _create_market_data_channel(
        self, product: InteractiveBrokersProduct, resolution: Resolution
    ):
        """Creates publich channel for internal use.
        Since IB's subscription does not require channel name,
        this function creates channel only for internal use, clarity and consistency.
        """
        self.add_product(product)
        metadata = self.venue.venue_class.METADATA
        supported_resolutions = cast(
            dict[Resolution | Timeframe, list[int]], metadata.supported_resolutions
        )
        if resolution.is_quote():
            if resolution not in supported_resolutions:
                raise ValueError(
                    f"{self.name} {product.symbol} {resolution=} is not supported"
                )
            channel = DataChannel.orderbook
            if resolution.orderbook_level == 2:
                if not self._config.reqMktDepthL2:
                    echannel = self.adapter(channel, group="channels")
                else:
                    echannel = "reqMktDepthL2"
            else:
                echannel = "reqMktData"
            orderbook_depth = resolution.period
            supported_orderbook_depths = supported_resolutions[resolution]
            if orderbook_depth not in supported_orderbook_depths:
                raise ValueError(
                    f"{self.name} {product.symbol} {orderbook_depth=} is not supported, {supported_orderbook_depths=}"
                )
            full_channel = ".".join([echannel, str(orderbook_depth), product.symbol])
        elif resolution.is_tick():
            channel = DataChannel.tradebook
            echannel = self.adapter(channel, group="channels")
            full_channel = ".".join([echannel, product.symbol])
        elif resolution.is_bar():
            channel = DataChannel.candlestick
            echannel = self.adapter(channel, group="channels")
            period = resolution.period
            if resolution.timeframe not in supported_resolutions:
                raise ValueError(
                    f"{self.venue} {product.symbol} {resolution=} is not supported, supported resolutions:\n"
                    + f"{pformat(list(supported_resolutions))}"
                )
            elif period not in supported_resolutions[resolution.timeframe]:
                raise ValueError(
                    f"{self.venue} {product.symbol} {resolution=} {period=} is not supported, supported periods\n:"
                    + f"{pformat(supported_resolutions[resolution.timeframe])}"
                )
            eresolution = self.adapter(repr(resolution), group="channel_resolutions")
            full_channel = ".".join([echannel, eresolution, product.symbol])
        else:
            raise NotImplementedError(
                f"{resolution=} is not supported for creating public channel"
            )
        return full_channel

    def _create_private_channel(
        self, channel: PrivateDataChannel | str
    ) -> FullDataChannel:
        return str(self.adapter(channel, group="channels"))

    def connect(self, account: InteractiveBrokersAccount):
        super().connect(
            host=account.host, port=account.port, clientId=account.client_id
        )  # pyright: ignore[reportUnknownMemberType]
        self._ib_thread.start()
        self._logger.debug(f"{self.venue} thread started")
        # TODO
        # if self._wait(self.is_connected, reason='connection'):
        #     # need to wait for the EReader to get ready; otherwise,
        #     # if subscribe too early and the subscription failed,
        #     # it will somehow lead to disconnection from IB
        #     time.sleep(1)
        #     self._subscribe()
        #     # wait for subscription
        #     self._background_thread = Thread(target=self._run_background_tasks, daemon=True)
        #     self._background_thread.start()

    def disconnect(self, reason: str = ""):
        self._logger.warning(f"{self.name} is disconnecting, {reason=}")
        super().disconnect()
        # self._unsubscribe()

    def _subscribe(self, channels: list[str], channel_type: DataChannelType):
        # TODO
        # ib_params = self._ib_params_for_channels_subscription[full_channel]
        for channel in channels:
            if channel_type == DataChannelType.public:
                pass
                # if channel == 'kline':
                #     self._request_real_time_bar(**ib_params)

                # if channel == 'orderbook':
                #     if self._orderbook_level[pdt] == 1:
                #         if product.ptype not in self.ASSET_TYPES_WITHOUT_TICK_BY_TICK_DATA:
                #             tick_type = ib_params.get('tickType', 'BidAsk')
                #             assert tick_type in ['MidPoint', 'BidAsk'], f'tickType={tick_type} is not supported for trade channel'
                #             self._request_tick_by_tick_data(tick_type, **ib_params)
                #         else:
                #             self._request_market_data(**ib_params)
                #             self._subscribed_market_data_tick_types[pdt].extend([TickTypeEnum.BID, TickTypeEnum.BID_SIZE, TickTypeEnum.ASK, TickTypeEnum.ASK_SIZE])
                #     elif self._orderbook_level[pdt] == 2:
                #         self._request_market_depth(**ib_params)
                # elif channel == 'tradebook':
                #     if product.ptype not in self.ASSET_TYPES_WITHOUT_TICK_BY_TICK_DATA + self.ASSET_TYPES_WITHOUT_TICK_BY_TICK_LAST_DATA:
                #         tick_type = ib_params.get('tickType', 'Last')
                #         assert tick_type in ['Last', 'AllLast'], f'tickType={tick_type} is not supported for trade channel'
                #         self._request_tick_by_tick_data(tick_type, **ib_params)
                #     else:
                #         self._request_market_data(**ib_params)
                #         self._subscribed_market_data_tick_types[pdt].extend([TickTypeEnum.LAST, TickTypeEnum.LAST_SIZE])

                # if did not request market data but defined related params for it, request for it anyways
                # if not self._subscribed_market_data_tick_types[pdt] and \
                #     any(params in ib_params for params in ['genericTickList', 'snapshot', 'regulatorySnapshot']):
                #     self._request_market_data(**ib_params)
            else:
                if channel == "account_update":
                    self._request_account_updates(acc)
                elif channel == "account_summary":
                    self._request_account_summary(**ib_params)

    def _unsubscribe(self):
        pass

    # def _update_orderbook(
    #     self, req_id, position: int, operation: int, side: int, px, qty, **kwargs
    # ):
    #     """
    #     Args:
    #         position: the orderbook's row being updated
    #         operation: 0 = insert, 1 = update, 2 = remove
    #         side: 0 = ask, 1 = bid
    #     """

    #     # boa = bids or asks
    #     def _update(boa: list):
    #         if operation == 0:
    #             boa.insert(position, (Decimal(px), qty))
    #         elif operation == 1:
    #             boa[position] = (Decimal(px), qty)
    #         elif operation == 2:
    #             del boa[position]

    #     try:
    #         pdt = self._product_by_req_id[req_id]
    #         if side == 0:
    #             bids, asks = None, self._asks[pdt]
    #             _update(asks)
    #         else:
    #             bids, asks = self._bids[pdt], None
    #             _update(bids)
    #         zmq_msg = (
    #             1,
    #             1,
    #             (self._bkr, product.exchange, str(product), bids, asks, None, kwargs),
    #         )
    #         self._zmq.send(*zmq_msg, receiver="engine")
    #     except:
    #         self._logger.exception(
    #             f"_update_orderbook exception ({position=} {operation=} {side=} {px=} {qty=} {kwargs=}):"
    #         )
