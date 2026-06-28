from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, Generic, ClassVar, cast, TypeVar

if TYPE_CHECKING:
    from http import HTTPMethod
    from httpx2._types import QueryParamTypes, RequestData
    from pfund.typing import FullDataChannel
    from pfund.datas.resolution import Resolution
    from pfund.venues.venue_config import VenueConfig
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.venues._apis.rest_api_base import BaseRestAPI
    from pfund.venues._apis.ws_api_base import BaseWebSocketAPI

import time
from abc import ABC, abstractmethod

from pfund.venues.venue_base import (
    BaseVenue,
    ConfigT,
    MarketT,
    AccountT,
    BalanceT,
    OrderT,
    ProductT,
    PositionT,
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
    BaseVenue[ConfigT, MarketT, AccountT, BalanceT, OrderT, ProductT, PositionT],
    Generic[
        RestAPITypeVar,
        WebSocketAPITypeVar,
        ConfigT,
        MarketT,
        AccountT,
        BalanceT,
        OrderT,
        ProductT,
        PositionT,
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
        super().__init__(env=env, config=cast(ConfigT, config), settings=settings)
        self.rest_api = cast("RestAPITypeVar", self.RestAPI(env=self.env))
        self.ws_api = cast("WebSocketAPITypeVar", self.WebSocketAPI(env=self.env))

    def _add_product(self, product: ProductT) -> None:
        super()._add_product(product)
        self.ws_api._add_product(product)

    def _add_account(self, account: AccountT) -> None:
        super()._add_account(account)
        self.ws_api._add_account(account)

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

    # # TODO: update to get rid of step_into()
    # def get_orders(
    #     self, account: CryptoAccount, schema, params=None, **kwargs
    # ) -> dict | None:
    #     orders = {
    #         "ts": None,
    #         "data": defaultdict(list),
    #         "source": OrderUpdateSource.GOO,
    #     }
    #     ret = self._rest_api.get_orders(account, params=params)
    #     res = self._parse_return(ret, schema["result"], default_result=False)
    #     res_type = type(res)
    #     if res:
    #         if "ts" in schema:
    #             orders["ts"] = float(step_into(ret, schema["ts"])) * schema["ts_adj"]
    #         if res_type is list:
    #             for order in res:
    #                 epdt = step_into(order, schema["pdt"])
    #                 category = params.get("category", "")
    #                 pdt = self.adapter(epdt, group=product.type)
    #                 update = {}
    #                 for k, (ek, *sequence) in schema["data"].items():
    #                     group = k + "s" if k in ["tif", "side"] else ""
    #                     initial_value = self.adapter(step_into(order, ek), group=group)
    #                     v = reduce(
    #                         lambda v, f: f(v) if v else v, sequence, initial_value
    #                     )
    #                     update[k] = v
    #                 orders["data"][pdt].append(update)
    #                 eoid = update["eoid"]
    #         else:
    #             raise Exception(f"{self.name} unhandled {res_type=}")
    #     if self._zmq and orders["data"]:
    #         zmq_msg = (2, 1, (self.bkr, self.name, account.acc, orders))
    #         self._zmq.send(*zmq_msg, receiver="engine")
    #     return orders

    # def get_balances(
    #     self, account: CryptoAccount, schema, params=None, **kwargs
    # ) -> dict | None:
    #     balances = {"ts": None, "data": defaultdict(dict)}
    #     ret = self._rest_api.get_balances(account, params=params)
    #     res = self._parse_return(ret, schema["result"], default_result=False)
    #     res_type = type(res)
    #     if res:
    #         if "ts" in schema:
    #             balances["ts"] = float(step_into(ret, schema["ts"])) * schema["ts_adj"]
    #         if res_type is dict:
    #             for eccy, balance in res.items():
    #                 ccy = self.adapter(eccy, group="assets")
    #                 for k, (ek, *sequence) in schema["data"].items():
    #                     initial_value = self.adapter(step_into(balance, ek))
    #                     v = reduce(
    #                         lambda v, f: f(v) if v else v, sequence, initial_value
    #                     )
    #                     balances["data"][ccy][k] = v
    #         elif res_type is list:
    #             for balance in res:
    #                 eccy = step_into(balance, schema["ccy"])
    #                 ccy = self.adapter(eccy, group="assets")
    #                 for k, (ek, *sequence) in schema["data"].items():
    #                     initial_value = self.adapter(step_into(balance, ek))
    #                     v = reduce(
    #                         lambda v, f: f(v) if v else v, sequence, initial_value
    #                     )
    #                     balances["data"][ccy][k] = v
    #         else:
    #             raise Exception(f"{self.name} unhandled {res_type=}")
    #     if self._zmq and balances["data"]:
    #         zmq_msg = (
    #             3,
    #             1,
    #             (
    #                 self.bkr,
    #                 self.name,
    #                 account.acc,
    #                 balances,
    #             ),
    #         )
    #         self._zmq.send(*zmq_msg, receiver="engine")
    #     return balances

    # def get_positions(
    #     self, account: CryptoAccount, schema, params=None, **kwargs
    # ) -> dict | None:
    #     from numpy import sign

    #     positions = {"ts": None, "data": defaultdict(dict)}
    #     ret = self._rest_api.get_positions(account, params=params)
    #     res = self._parse_return(ret, schema["result"], default_result=False)
    #     res_type = type(res)
    #     if res:
    #         if "ts" in schema:
    #             positions["ts"] = float(step_into(ret, schema["ts"])) * schema["ts_adj"]
    #         if res_type is list:
    #             for position in res:
    #                 epdt = step_into(position, schema["pdt"])
    #                 category = params.get("category", "")
    #                 # TODO: convert category to product asset type
    #                 pdt = self.adapter(epdt, group=asset_type)
    #                 qty = float(step_into(position, schema["data"]["qty"][0]))
    #                 if qty == 0 and pdt not in self._products:
    #                     continue
    #                 if "side" in schema:
    #                     eside = step_into(position, schema["side"])
    #                     side = self.adapter(eside, group="side")
    #                 # e.g. BINANCE_USDT only returns position size (signed qty)
    #                 elif "size" in schema:
    #                     side = sign(step_into(position, schema["size"]))
    #                 positions["data"][pdt][side] = {}
    #                 for k, (ek, *sequence) in schema["data"].items():
    #                     initial_value = self.adapter(step_into(position, ek))
    #                     v = reduce(
    #                         lambda v, f: f(v) if v else v, sequence, initial_value
    #                     )
    #                     positions["data"][pdt][side][k] = v
    #         else:
    #             raise Exception(f"{self.name} unhandled {res_type=}")
    #     if self._zmq and positions["data"]:
    #         zmq_msg = (
    #             3,
    #             2,
    #             (
    #                 self.bkr,
    #                 self.name,
    #                 account.acc,
    #                 positions,
    #             ),
    #         )
    #         self._zmq.send(*zmq_msg, receiver="engine")
    #     return positions

    # def get_trades(
    #     self, account: CryptoAccount, schema, params=None, **kwargs
    # ) -> dict | None:
    #     trades = {
    #         "ts": None,
    #         "data": defaultdict(list),
    #         "source": OrderUpdateSource.GTH,
    #     }
    #     ret = self._rest_api.get_trades(account, params=params)
    #     res = self._parse_return(ret, schema["result"], default_result=False)
    #     res_type = type(res)
    #     if res:
    #         if "ts" in schema:
    #             trades["ts"] = float(step_into(ret, schema["ts"])) * schema["ts_adj"]
    #         if res_type is list:
    #             for trade in res:
    #                 epdt = step_into(trade, schema["pdt"])
    #                 category = params.get("category", "")
    #                 # TODO: convert category to product asset type
    #                 pdt = self.adapter(epdt, group=asset_type)
    #                 update = {}
    #                 for k, (ek, *sequence) in schema["data"].items():
    #                     group = k + "s" if k in ["tif", "side"] else ""
    #                     initial_value = self.adapter(step_into(trade, ek), group=group)
    #                     v = reduce(
    #                         lambda v, f: f(v) if v else v, sequence, initial_value
    #                     )
    #                     update[k] = v
    #                     if k == "trade_ts":
    #                         update[k] *= schema["ts_adj"]
    #                 trades["data"][pdt].append(update)
    #         else:
    #             raise Exception(f"{self.name} unhandled {res_type=}")
    #     if self._zmq and trades["data"]:
    #         zmq_msg = (2, 1, (self.bkr, self.name, account.acc, trades))
    #         self._zmq.send(*zmq_msg, receiver="engine")
    #     return trades

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
