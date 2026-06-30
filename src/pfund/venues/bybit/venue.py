from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from pfund.venues._apis.typing import Result

from pfund.typing import ProductKey
from pfund.datas.timeframe import Timeframe
from pfund.venues.crypto_exchange import CryptoExchange
from pfund.venues.bybit.rest_api import BybitRestAPI
from pfund.venues.bybit.ws_api import BybitWebSocketAPI
from pfund.venues.venue_metadata import VenueMetadata
from pfund.venues.bybit.adapter import BybitAdapter
from pfund.venues.bybit.config import BybitConfig
from pfund.venues.bybit.market import BybitMarket
from pfund.venues.bybit.account import BybitAccount
from pfund.venues.bybit.balance import BybitBalance
from pfund.venues.bybit.order import BybitOrder
from pfund.venues.bybit.product import BybitProduct
from pfund.venues.bybit.position import BybitPosition
from pfund.entities.products.asset_type import AssetType
from pfund.enums import AssetTypeModifier, CryptoAssetType, TradingVenue


class Bybit(
    CryptoExchange[
        BybitRestAPI,
        BybitWebSocketAPI,
        BybitConfig,
        BybitMarket,
        BybitAccount,
        BybitBalance,
        BybitOrder,
        BybitProduct,
        BybitPosition,
    ]
):
    name: ClassVar[TradingVenue] = TradingVenue.BYBIT
    adapter: ClassVar[BybitAdapter] = BybitAdapter()

    RestAPI: ClassVar[type[BybitRestAPI]] = BybitRestAPI
    WebSocketAPI: ClassVar[type[BybitWebSocketAPI]] = BybitWebSocketAPI

    Config: ClassVar[type[BybitConfig]] = BybitConfig
    Market: ClassVar[type[BybitMarket]] = BybitMarket
    Account: ClassVar[type[BybitAccount]] = BybitAccount
    Balance: ClassVar[type[BybitBalance]] = BybitBalance
    Order: ClassVar[type[BybitOrder]] = BybitOrder
    Product: ClassVar[type[BybitProduct]] = BybitProduct
    Position: ClassVar[type[BybitPosition]] = BybitPosition

    METADATA: ClassVar[VenueMetadata] = VenueMetadata(
        requires_asyncio_loop=True,
        asset_types=[
            CryptoAssetType.FUTURE,
            CryptoAssetType.PERPETUAL,
            CryptoAssetType.OPTION,
            CryptoAssetType.CRYPTO,
            CryptoAssetType.INDEX,
            AssetTypeModifier.INVERSE + "-" + CryptoAssetType.FUTURE,
            AssetTypeModifier.INVERSE + "-" + CryptoAssetType.PERPETUAL,
        ],
        stream_resolution_periods={
            BybitProduct.Category.LINEAR: {
                Timeframe.QUOTE: [1, 50, 200, 500],
                Timeframe.TICK: [1],
                Timeframe.MINUTE: [1, 3, 5, 15, 30, 60, 120, 240, 360, 720],
                Timeframe.DAY: [1],
            },
            BybitProduct.Category.INVERSE: {
                Timeframe.QUOTE: [1, 50, 200, 500],
                Timeframe.TICK: [1],
                Timeframe.MINUTE: [1, 3, 5, 15, 30, 60, 120, 240, 360, 720],
                Timeframe.DAY: [1],
            },
            BybitProduct.Category.SPOT: {
                Timeframe.QUOTE: [1, 50],
                Timeframe.TICK: [1],
                Timeframe.MINUTE: [1, 3, 5, 15, 30, 60, 120, 240, 360, 720],
                Timeframe.DAY: [1],
            },
            BybitProduct.Category.OPTION: {
                Timeframe.QUOTE: [25, 100],
                Timeframe.TICK: [1],
                Timeframe.MINUTE: [1, 3, 5, 15, 30, 60, 120, 240, 360, 720],
                Timeframe.DAY: [1],
            },
        },
        stream_orderbook_levels={
            BybitProduct.Category.LINEAR: [1, 2],
            BybitProduct.Category.INVERSE: [1, 2],
            BybitProduct.Category.SPOT: [1, 2],
            BybitProduct.Category.OPTION: [2],
        },
        support_place_batch_orders=True,
        support_cancel_batch_orders=True,
        support_amend_batch_orders=True,
    )

    async def get_markets_async(
        self,
        category: BybitProduct.Category
        | Literal["LINEAR", "OPTION", "SPOT", "OPTION"]
        | None = None,
    ) -> dict[ProductKey, BybitMarket]:
        if category is None:
            categories = [category for category in BybitProduct.Category]
        else:
            categories = [BybitProduct.Category[category.upper()]]

        async def _fetch_category(
            category: BybitProduct.Category,
        ) -> dict[ProductKey, BybitMarket]:
            result: Result = await self.rest_api.get_markets(category=category)
            data = result["response"]["data"]
            if not isinstance(data, list):
                raise RuntimeError(
                    f"unexpected data for category {category}: "
                    + f"expected list, got {type(data).__name__}"
                )
            markets: dict[ProductKey, BybitMarket] = {}
            for data_per_market in data:
                market = BybitMarket(**data_per_market)
                product_key = ProductKey(
                    symbol=market.symbol, asset_type=AssetType(value=market.asset_type)
                )
                markets[product_key] = market
            return markets

        # fan out one request per category and await them together
        results = await asyncio.gather(*(_fetch_category(c) for c in categories))

        markets: dict[ProductKey, BybitMarket] = {}
        for markets_per_category in results:
            markets.update(markets_per_category)
        return markets

    aget_markets = get_markets_async

    def get_markets(
        self,
        category: BybitProduct.Category
        | Literal["LINEAR", "OPTION", "SPOT", "OPTION"]
        | None = None,
    ) -> dict[ProductKey, BybitMarket]:
        return self._run_async(self.aget_markets(category=category))

    # def get_balances(
    #     self, account: CryptoAccount, ccy: str = "", **kwargs
    # ) -> dict[str, dict]:
    #     schema = {
    #         # result->list will return a useless list type containing a dict,
    #         # need index '0' to get the real result
    #         # TODO, need to make sure it has really only one result so that using index 0 is safe
    #         "@data": ["result", "list", 0, "coin"],
    #         "ts": "time",
    #         "ts_adj": 1 / 10**3,  # since timestamp in bybit is in mts
    #         "ccy": "coin",
    #         "data": {
    #             "wallet": ("walletBalance", str, Decimal),
    #             "available": ("availableToWithdraw", str, Decimal),
    #             "margin": ("equity", str, Decimal),
    #         },
    #     }
    #     params = {"accountType": account.type}
    #     if ccy:
    #         params["coin"] = self.adapter(ccy)
    #     if kwargs:
    #         params.update(kwargs)
    #     return super().get_balances(
    #         account,
    #         schema,
    #         params=params,
    #     )

    # # FIXME: remove pdt, pass in product object
    # def get_positions(
    #     self,
    #     account: CryptoAccount,
    #     pdt: str = "",
    #     category: tProductCategory = "",
    #     **kwargs,
    # ) -> dict | None:
    #     schema = {
    #         "@data": ["result", "list"],
    #         "ts": "time",
    #         "ts_adj": 1 / 10**3,  # since timestamp in bybit is in mts
    #         "pdt": "symbol",
    #         "side": "side",
    #         "data": {
    #             "qty": ("size", str, Decimal, abs),
    #             "avg_px": ("avgPrice", str, Decimal),
    #             "liquidation_px": ("liqPrice", str, Decimal),
    #             "unrealized_pnl": ("unrealisedPnl", str, Decimal),
    #             "realized_pnl": ("cumRealisedPnl", str, Decimal),
    #         },
    #     }
    #     products = [self.get_product(pdt)] if pdt else list(self.products.values())
    #     # FIXME: use pydantic model PositionUpdate(), not dict
    #     positions = {"ts": 0.0, "data": {}}
    #     categories = [category] if category else self._categories
    #     products_per_category = {
    #         category: [product for product in products if product.category == category]
    #         for category in categories
    #     }
    #     for category, products in products_per_category.items():
    #         if pdt:
    #             iterator = pdts = set(str(product) for product in products)
    #         else:
    #             iterator = qccys = set(product.qccy for product in products)
    #         for element in iterator:
    #             params = {"category": category}
    #             if pdt:
    #                 epdt = self.adapter(element, group=product.type)
    #                 params["symbol"] = epdt
    #             else:
    #                 eqccy = self.adapter(element)
    #                 params["settleCoin"] = eqccy
    #             if kwargs:
    #                 params.update(kwargs)
    #             categorized_positions = super().get_positions(
    #                 account,
    #                 schema,
    #                 params=params,
    #             )
    #             if categorized_positions:
    #                 if categorized_positions["ts"]:
    #                     positions["ts"] = max(
    #                         positions["ts"], categorized_positions["ts"]
    #                     )
    #                 if categorized_positions["data"]:
    #                     positions["data"].update(categorized_positions["data"])
    #             else:
    #                 positions = categorized_positions
    #     return positions

    # def get_orders(
    #     self,
    #     account: CryptoAccount,
    #     pdt: str = "",
    #     category: tProductCategory = "",
    #     **kwargs,
    # ):
    #     schema = {
    #         "@data": ["result", "list"],
    #         "ts": "time",
    #         "ts_adj": 1 / 10**3,  # since timestamp in bybit is in mts
    #         "pdt": "symbol",
    #         "data": {
    #             "oid": ("orderLinkId", str),
    #             "eoid": ("orderId", str),
    #             "side": ("side", int),
    #             "px": ("price", str, Decimal),
    #             "qty": ("qty", str, Decimal, abs),
    #             "avg_px": ("avgPrice", str, Decimal),
    #             "filled_qty": ("cumExecQty", str, Decimal, abs),
    #             # FIXME (not sure) price that triggers a stop loss/take profit order
    #             "trigger_px": ("triggerPrice", str, Decimal),
    #             "o_type": ("orderType", str),
    #             "status": ("orderStatus", str),
    #             "tif": ("timeInForce", str),
    #             "is_reduce_only": ("reduceOnly", bool),
    #         },
    #     }
    #     products = [self.get_product(pdt)] if pdt else list(self.products.values())
    #     orders = {"ts": 0.0, "data": {}, "source": None}
    #     categories = [category] if category else self._categories
    #     products_per_category = {
    #         category: [product for product in products if product.category == category]
    #         for category in categories
    #     }
    #     for category, products in products_per_category.items():
    #         if pdt:
    #             iterator = pdts = set(str(product) for product in products)
    #         else:
    #             iterator = qccys = set(product.qccy for product in products)
    #         for element in iterator:
    #             params = {"category": category}
    #             if pdt:
    #                 epdt = self.adapter(element, group=product.type)
    #                 params["symbol"] = epdt
    #             else:
    #                 eqccy = self.adapter(element)
    #                 params["settleCoin"] = eqccy
    #             if kwargs:
    #                 params.update(kwargs)
    #             categorized_orders = super().get_orders(
    #                 account,
    #                 schema,
    #                 params=params,
    #             )
    #             if categorized_orders:
    #                 if categorized_orders["ts"]:
    #                     orders["ts"] = max(orders["ts"], categorized_orders["ts"])
    #                 if categorized_orders["data"]:
    #                     orders["data"].update(categorized_orders["data"])
    #                 orders["source"] = categorized_orders["source"]
    #             else:
    #                 orders = categorized_orders
    #     return orders

    # def get_trades(
    #     self,
    #     account: CryptoAccount,
    #     pdt: str = "",
    #     category: tProductCategory = "",
    #     start_time: str | float = None,
    #     end_time: str | float = None,
    #     is_funding_considered_as_trades=False,
    #     **kwargs,
    # ):
    #     """
    #     Args:
    #         start_time: start time of trade history,
    #             if datetime (string) in UTC is provided, supported format is '%Y-%m-%d %H:%M:%S'
    #             if timestamp (float) is provided, unit should be in seconds
    #         end_time: end time of trade history,
    #             if datetime (string) in UTC is provided, supported format is '%Y-%m-%d %H:%M:%S'
    #             if timestamp (float) is provided, unit should be in seconds
    #     """

    #     def _convert_to_date(time_):
    #         if type(time_) is float:
    #             date = datetime.datetime.fromtimestamp(time_, tz=datetime.UTC)
    #         elif type(time_) is str:
    #             date = datetime.datetime.strptime(time_, date_format)
    #             date = date.replace(tzinfo=datetime.UTC)
    #         return date

    #     schema = {
    #         "@data": ["result", "list"],
    #         "ts": "time",
    #         "ts_adj": 1 / 10**3,  # since timestamp in bybit is in mts
    #         "pdt": "symbol",
    #         "data": {
    #             "oid": ("orderLinkId", str),
    #             "eoid": ("orderId", str),
    #             "side": ("side", int),
    #             "px": ("orderPrice", str, Decimal),
    #             "qty": ("orderQty", str, Decimal, abs),
    #             "ltp": ("execPrice", str, Decimal),
    #             "ltq": ("execQty", str, Decimal, abs),
    #             "o_type": ("orderType", str),
    #             "trade_ts": ("execTime", float),
    #             # 'trade_id': ('execId', str),
    #             # specific to bybit
    #             "trade_type": ("execType", str),
    #         },
    #     }

    #     default_rollback_hours = 1
    #     date_format = "%Y-%m-%d %H:%M:%S"
    #     end_date = (
    #         datetime.datetime.now(tz=datetime.UTC)
    #         if end_time is None
    #         else _convert_to_date(end_time)
    #     )
    #     start_date = (
    #         end_date - datetime.timedelta(hours=default_rollback_hours)
    #         if start_time is None
    #         else _convert_to_date(start_time)
    #     )
    #     end_time = int(end_date.timestamp() * 1000)  # bybit requires mts
    #     start_time = int(start_date.timestamp() * 1000)  # bybit requires mts

    #     trades = {"ts": 0.0, "data": {}, "source": None}
    #     categories = [category] if category else self._categories
    #     for category in categories:
    #         params = {
    #             "category": category,
    #             "startTime": start_time,
    #             "endTime": end_time,
    #         }
    #         if pdt:
    #             epdt = self.adapter(pdt, group=product.type)
    #             params["symbol"] = epdt
    #         if kwargs:
    #             params.update(kwargs)
    #         categorized_trades = super().get_trades(
    #             account,
    #             schema,
    #             params=params,
    #         )
    #         if categorized_trades:
    #             if categorized_trades["ts"]:
    #                 trades["ts"] = max(trades["ts"], categorized_trades["ts"])
    #             if categorized_trades["data"]:
    #                 trades["data"].update(categorized_trades["data"])
    #             trades["source"] = categorized_trades["source"]

    #             # specific to bybit, remove all the 'Funding' trades
    #             if not is_funding_considered_as_trades:
    #                 for pdt in trades["data"]:
    #                     for trade in trades["data"][pdt][:]:
    #                         if trade["trade_type"] != "Trade":
    #                             trades["data"][pdt].remove(trade)
    #                         else:
    #                             del trade["trade_type"]

    #             for pdt in trades["data"]:
    #                 trades["data"][pdt] = self._combine_trades(trades["data"][pdt])
    #         else:
    #             trades = categorized_trades
    #     return trades

    # def place_order(
    #     self,
    #     account: CryptoAccount,
    #     product: BybitProduct,
    #     order: BaseOrder,
    #     expires_in: int = 5000,
    # ):
    #     """
    #     Args:
    #         expires_in: time in milliseconds, specify how long the HTTP request is valid.
    #     """
    #     schema = {
    #         "@data": "result",
    #         "ts": "time",
    #         "ts_adj": 1 / 10**3,  # convert bybit's milliseconds to seconds
    #         "data": {
    #             "oid": ("orderLinkId", str),
    #             "eoid": ("orderId", str),
    #         },
    #     }
    #     params = {
    #         "category": product.category,
    #         "symbol": self.adapter(order.pdt, group=product.category),
    #         "side": self.adapter(order.side, group="sides"),
    #         "orderType": self.adapter(order.type),
    #         "qty": str(order.qty),
    #         "timeInForce": order.tif,
    #         "orderLinkId": order.oid,
    #     }
    #     if order.px:
    #         params["price"] = str(order.px)
    #     # REVIEW, maybe create a class BybitOrder to better handle this?
    #     if hasattr(order, "isLeverage"):
    #         params["isLeverage"] = order.isLeverage
    #     if hasattr(order, "orderFilter"):
    #         params["orderFilter"] = order.orderFilter
    #     if hasattr(order, "orderLv"):
    #         params["orderLv"] = order.orderLv
    #     if hasattr(order, "positionIdx"):
    #         params["positionIdx"] = int(order.positionIdx)
    #     if hasattr(order, "closeOnTrigger"):
    #         params["closeOnTrigger"] = order.closeOnTrigger
    #     if hasattr(order, "mmp"):
    #         params["mmp"] = order.mmp
    #     if hasattr(order, "smpType"):
    #         params["smpType"] = order.smpType

    #     update = super().place_order(
    #         account, schema, params=params, expires_in=expires_in
    #     )
    #     # bybit's return has no order status, create it manually
    #     update["status"] = "O---"
    #     return update

    # def cancel_order(
    #     self, account: CryptoAccount, product: BybitProduct, order: BaseOrder
    # ):
    #     schema = {
    #         "@data": "result",
    #         "ts": "time",
    #         "ts_adj": 1 / 10**3,  # since timestamp in bybit is in mts
    #         "data": {
    #             "oid": ("orderLinkId", str),
    #             "eoid": ("orderId", str),
    #         },
    #     }
    #     params = {
    #         "category": product.category,
    #         "symbol": self.adapter(order.pdt, group=product.category),
    #         "orderLinkId": order.oid,
    #         "orderId": order.eoid,
    #     }
    #     # REVIEW, maybe create a class BybitOrder to better handle this?
    #     if hasattr(order, "orderFilter"):
    #         params["orderFilter"] = order.orderFilter
    #     update = super().cancel_order(account, schema, params=params)
    #     # bybit's return has no order status, create it manually
    #     update["status"] = "C-C-"
    #     return update

    # # NOTE, bybit only supports place_batch_orders for category `options`
    # # TODO, come back to this if bybit supports more
    # def place_batch_orders(
    #     self, account: CryptoAccount, product: BybitProduct, orders: list[BaseOrder]
    # ):
    #     assert len(orders) <= self.MAX_NUM_OF_PLACE_BATCH_ORDERS

    # # NOTE, bybit only supports cancel_batch_orders for category `options`
    # # TODO, come back to this if bybit supports more
    # def cancel_batch_orders(
    #     self, account: CryptoAccount, product: BybitProduct, orders: list[BaseOrder]
    # ):
    #     assert len(orders) <= self._MAX_NUM_OF_CANCEL_BATCH_ORDERS
