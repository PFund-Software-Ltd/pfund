from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from pathlib import Path

    from pfund.venues._apis.rest_api_base import Result
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.entities.accounts import APIKeyAccount
    from pfund.typing import AccountName, FullDataChannel

    ProductCategory: TypeAlias = str


import datetime
import os
from abc import abstractmethod
from collections import defaultdict
from functools import reduce

from pfund.enums import DataChannel
from pfund.venues.venue_base import BaseVenue


class CryptoExchange(BaseVenue):
    accounts: dict[AccountName, APIKeyAccount]

    # @classmethod
    # def create_product(
    #     cls, basis: str, name: str = "", symbol: str = "", **specs: Any
    # ) -> CryptoProduct:
    #     from pfeed.enums import DataSource

    #     from pfund.entities.products import ProductFactory

    #     source = DataSource[cls.NAME.upper()]
    #     Product = ProductFactory(source=source, basis=basis)
    #     return Product(basis=basis, name=name, symbol=symbol, specs=specs)

    # def get_product(self, name: str) -> CryptoProduct:
    #     """
    #     Args:
    #         name: product name (product.name)
    #     """
    #     return self._products[name]

    # def get_account(self, name: str) -> CryptoAccount:
    #     return self._accounts[name]

    # def add_account(
    #     self,
    #     venue: TradingVenue,
    #     name: AccountName = "",
    #     key: str = "",
    #     secret: str = "",
    # ) -> CryptoAccount:
    #     exch = CryptoExchange[venue.upper()]
    #     exchange = self.add_exchange(exch)
    #     if name not in self._accounts[exchange.name]:
    #         account = CryptoAccount(
    #             env=self._env, exchange=exch, name=name, key=key, secret=secret
    #         )
    #         if account.name not in self._accounts:
    #             self._accounts[account.name] = account
    #             self._ws_api.add_account(account)
    #         else:
    #             raise ValueError(f"account name {account.name} has already been added")
    #         self._accounts[exchange.name][account.name] = account
    #     else:
    #         raise ValueError(f"account name {name} has already been added")
    #     return account

    # def add_product(
    #     self,
    #     exch: CryptoExchange,
    #     basis: str,
    #     name: ProductName = "",
    #     symbol: str = "",
    #     **specs: Any,
    # ) -> CryptoProduct:
    #     """
    #     Args:
    #         name: product name (product.name)
    #     """
    #     exchange: BaseExchange = self.add_exchange(exch)
    #     # create another product object to get a correct product name
    #     product: CryptoProduct = exchange.create_product(
    #         basis, name=name, symbol=symbol, **specs
    #     )
    #     if product.name not in self._products[exchange.name]:
    #         if product.name not in self._products:
    #             self._products[product.name] = product
    #             self._ws_api.add_product(product)
    #             self.adapter.add_mapping(str(product.type), product.name, product.symbol)
    #         else:
    #             existing_product: CryptoProduct = self.get_product(product.name)
    #             # assert products are the same with the same name
    #             if existing_product == product:
    #                 product = existing_product
    #             else:
    #                 raise ValueError(
    #                     f"product name {product.name} has already been used for {existing_product}"
    #                 )
    #         self._products[exchange.name][product.name] = product
    #     else:
    #         existing_product: CryptoProduct = self.get_product(exch, product.name)
    #         # assert products are the same with the same name
    #         if existing_product == product:
    #             product = existing_product
    #         else:
    #             raise ValueError(
    #                 f"product name {name} has already been used for {existing_product}"
    #             )
    #     return product

    # def add_public_channel(
    #     self,
    #     channel: DataChannel | FullDataChannel,
    #     data: TimeBasedData | None = None,
    # ):
    #     if channel.lower() in DataChannel.__members__:
    #         assert data is not None, "data object is required for public channels"
    #         channel: FullDataChannel = self._ws_api._create_public_channel(
    #             data.product, data.resolution
    #         )
    #     self._ws_api.add_channel(channel, channel_type="public")

    # def add_private_channel(self, channel: PrivateDataChannel | FullDataChannel):
    #     if channel.lower() in PrivateDataChannel.__members__:
    #         channel: FullDataChannel = self._ws_api._create_private_channel(channel)
    #     self._ws_api.add_channel(channel, channel_type="private")

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

    # """
    # ************************************************
    # API Calls
    # ************************************************
    # """

    # @abstractmethod
    # async def aget_markets(self, *args, **kwargs):
    #     pass

    # @abstractmethod
    # def get_markets(self, *args, **kwargs):
    #     pass

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
