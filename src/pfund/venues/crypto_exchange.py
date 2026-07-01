from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, Generic, ClassVar, cast, TypeVar


if TYPE_CHECKING:
    from http import HTTPMethod
    from httpx2._types import QueryParamTypes, RequestData
    from pfund.typing import FullDataChannel
    from pfund.venues._apis.typing import Result
    from pfund.datas.resolution import Resolution
    from pfund.venues.venue_config import VenueConfig
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.venues._apis.rest_api_base import BaseRestAPI
    from pfund.venues._apis.ws_api_base import BaseWebSocketAPI

import time
import queue
import asyncio
from abc import ABC, abstractmethod

from pfund.venues.venue_base import (
    BaseVenue,
    ConfigT,
    MarketT,
    AccountT,
    BalanceT,
    BalanceSnapshotT,
    OrderT,
    ProductT,
    PositionT,
    PositionSnapshotT,
)
from pfund.enums import Environment, PrivateDataChannel


RestAPITypeVar = TypeVar("RestAPITypeVar", bound="BaseRestAPI")
WebSocketAPITypeVar = TypeVar("WebSocketAPITypeVar", bound="BaseWebSocketAPI[Any, Any]")


class CryptoExchangeSigner(ABC, Generic[AccountT]):
    @property
    def nonce(self) -> int:
        return int(time.time() * 1000)

    @abstractmethod
    def sign_rest_api(
        self,
        account: AccountT,
        method: HTTPMethod,
        url: str,
        *,
        params: QueryParamTypes | None = None,
        json: Any | None = None,
        data: RequestData | None = None,
        headers: dict[str, str],
    ) -> None: ...

    @abstractmethod
    def sign_ws_api(self, account: AccountT) -> Any: ...


class CryptoExchange(
    BaseVenue[
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
    Generic[
        RestAPITypeVar,
        WebSocketAPITypeVar,
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
    """Crypto exchange that follows the standard pattern of using REST and WebSocket APIs."""

    RestAPI: ClassVar[type[BaseRestAPI]]
    WebSocketAPI: ClassVar[type[BaseWebSocketAPI[Any, Any]]]

    def __init__(
        self,
        env: Literal[Environment.PAPER, Environment.LIVE, "PAPER", "LIVE"],
        config: VenueConfig | None = None,
        settings: TradeEngineSettings | None = None,
    ):
        if config and type(config) is not self.Config:
            raise ValueError(f"config must be of type {self.Config}")
        if not self.METADATA.requires_asyncio_loop:
            raise ValueError(
                f"{self.name} requires an asyncio loop, did you forget to set requires_asyncio_loop=True in METADATA?"
            )
        super().__init__(env=env, config=cast(ConfigT, config), settings=settings)
        self.rest_api = cast("RestAPITypeVar", self.RestAPI(env=self.env))
        self.ws_api = cast("WebSocketAPITypeVar", self.WebSocketAPI(env=self.env))

    def _set_queue(self, queue: queue.Queue[Any]) -> None:
        super()._set_queue(queue)
        self.ws_api._set_queue(queue)

    def add_product(self, product: ProductT) -> None:
        super().add_product(product)
        self.ws_api.add_product(product)

    def add_account(self, account: AccountT) -> None:
        super().add_account(account)
        self.ws_api.add_account(account)

    def add_channel(
        self,
        channel: FullDataChannel,
        *,
        channel_type: Literal["public", "private"] = "public",
    ) -> None:
        self.ws_api.add_channel(channel, channel_type=channel_type)

    def _create_market_data_channel(
        self, product: ProductT, resolution: Resolution
    ) -> FullDataChannel:
        return self.ws_api._create_market_data_channel(product, resolution)

    def _create_private_channel(self, channel: PrivateDataChannel) -> FullDataChannel:
        return self.ws_api._create_private_channel(channel)

    async def _get_balances(self, account: AccountT) -> Result:
        return await self.rest_api.get_balances(account)

    # TODO:
    def place_orders(self, orders: list[OrderT]) -> None:
        pass
        # asyncio.run_coroutine_threadsafe(self.rest_api.function(), self.loop)
        #

    def start(self):
        super().start()
        self._run_coroutine_threadsafe(self.ws_api.connect)

    def stop(self):
        if self._loop and self._loop_thread and self._loop_thread.is_alive():
            future = asyncio.run_coroutine_threadsafe(
                self.ws_api.disconnect(reason="venue stopped"), self._loop
            )
            try:
                future.result(timeout=10)
            except Exception:
                self._logger.exception(f"{self.name} disconnect() failed during stop")
        super().stop()

    # # REVIEW
    # @staticmethod
    # def _combine_trades(trades: list):
    #     """
    #     Combines trades with the same eoid from trade history
    #     because some exchanges separate trades for the same order
    #     """
    #     trades_per_eoid = defaultdict(list)
    #     trades_combined = []
    #     for trade in trades:
    #         eoid = trade["eoid"]
    #         trades_per_eoid[eoid].append(trade)
    #     for trades in trades_per_eoid.values():
    #         # if multiple trades for the same order, combine them
    #         if len(trades) > 1:
    #             avg_px = filled_qty = 0.0
    #             trade_ts = 0.0
    #             for trade in trades:
    #                 last_traded_px, last_traded_qty = trade["ltp"], trade["ltq"]
    #                 avg_px += last_traded_px * last_traded_qty
    #                 filled_qty += last_traded_qty
    #                 trade_ts = max(trade_ts, trade["trade_ts"])
    #             avg_px /= filled_qty
    #             trade_adj = trades[-1]
    #             trade_adj["avg_px"] = avg_px
    #             trade_adj["filled_qty"] = filled_qty
    #         else:
    #             trade_adj = trades[0]
    #             trade_adj["avg_px"] = trade_adj["ltp"]
    #             trade_adj["filled_qty"] = trade_adj["ltq"]
    #         trades_combined.append(trade_adj)
    #     return trades_combined

    # def place_order(
    #     self, account: CryptoAccount, schema: dict, params=None, expires_in=5000
    # ):
    #     order = {"ts": None, "data": {}, "source": OrderUpdateSource.REST}
    #     ret = self._rest_api.place_order(account, params=params)
    #     res = self._parse_return(ret, schema["result"], default_result=False)
    #     res_type = type(res)
    #     if res:
    #         if "ts" in schema:
    #             order["ts"] = float(step_into(ret, schema["ts"])) * schema["ts_adj"]
    #         if res_type is dict:
    #             for k, (ek, *sequence) in schema["data"].items():
    #                 group = k + "s" if k in ["tif", "side"] else ""
    #                 initial_value = self.adapter(step_into(res, ek), group=group)
    #                 v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
    #                 order[k] = v
    #         else:
    #             raise Exception(f"{self.name} unhandled {res_type=}")
    #     if self._zmq and order["data"]:
    #         zmq_msg = (2, 1, (self.bkr, self.name, account.acc, order))
    #         self._zmq.send(*zmq_msg, receiver="engine")
    #     return order

    # def cancel_order(self, account: CryptoAccount, schema: dict, params=None, **kwargs):
    #     order = {"ts": None, "data": {}, "source": OrderUpdateSource.REST}
    #     ret = self._rest_api.cancel_order(account, params=params)
    #     res = self._parse_return(ret, schema["result"], default_result=False)
    #     res_type = type(res)
    #     if res:
    #         if "ts" in schema:
    #             order["ts"] = float(step_into(ret, schema["ts"])) * schema["ts_adj"]
    #         if res_type is dict:
    #             for k, (ek, *sequence) in schema["data"].items():
    #                 group = k + "s" if k in ["tif", "side"] else ""
    #                 initial_value = self.adapter(step_into(res, ek), group=group)
    #                 v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
    #                 order[k] = v
    #         else:
    #             raise Exception(f"{self.name} unhandled {res_type=}")
    #     if self._zmq and order["data"]:
    #         zmq_msg = (2, 1, (self.bkr, self.name, account.acc, order))
    #         self._zmq.send(*zmq_msg, receiver="engine")
    #     return order
