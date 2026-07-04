# pyright: reportUninitializedInstanceVariable=false
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Any, TypeAlias

if TYPE_CHECKING:
    from logging import Logger
    from pfund.venues.ibkr.product import InteractiveBrokersProduct

    RequestId: TypeAlias = int

import asyncio
from collections import defaultdict

from ibapi.account_summary_tags import *
from ibapi.client import EClient

from pfund.enums import TradingVenue


class InteractiveBrokersClient(EClient):
    _request_id: ClassVar[int] = 1

    venue: TradingVenue
    _logger: Logger

    def __init__(self):
        # pass in InteractiveBrokersAPI() object (child of EWrapper) as EClient needs EWrapper
        super().__init__(self)
        self._pending_responses: dict[int, asyncio.Future[list[Any]]] = {}
        # partial responses = accumulation buffer for multi-row replies.
        # An IB "response" is usually not one callback — it's a stream of callbacks terminated by an *End.
        self._partial_responses: dict[int, list[Any]] = defaultdict(list)

        self._product_by_req_id: dict[RequestId, InteractiveBrokersProduct] = {}
        self._pdts_requested_market_data = []

    @classmethod
    def _next_request_id(cls) -> int:
        cls._request_id += 1
        return cls._request_id

    def _update_request_id_and_corresponding_product(self, product):
        self._product_by_req_id[self._request_id] = product
        self._next_request_id()

    """
    public channels
    ---------------------------------------------------
    """

    def _request_market_data(self, **kwargs):
        """Aggregated level 1 orderbook + trade data (slower than tick by tick data)"""
        product = kwargs["product"]
        # market data subscription is for both bid/ask and last price/qty
        # if e.g. the 'orderbook' channel has already requested market data,
        # do not request again for the 'tradebook' channel
        if str(product) in self._pdts_requested_market_data:
            self._logger.debug(
                f"{self._bkr} has already requested {product!s} market data, do not request again"
            )
            return
        self.reqMktData(
            self._request_id,
            product,
            # generic_tick_list, snapshot, regulatory_snapshot are params in IB's reqMktData(...)
            kwargs.get("genericTickList", ""),
            kwargs.get("snapshot", False),
            kwargs.get("regulatorySnapshot", False),
            [],
        )
        self._logger.debug(
            f"{self._bkr} requested (req_id={self._request_id}) {product!s} market data"
        )
        self._pdts_requested_market_data.append(str(product))
        self._update_request_id_and_corresponding_product(product)

    def _request_tick_by_tick_data(
        self, tick_type: Literal["Last", "AllLast", "BidAsk", "MidPoint"], **kwargs
    ):
        """Level 1 orderbook/Trade data/Mid-point data"""
        product = kwargs["product"]
        self.reqTickByTickData(
            self._request_id,
            product,
            tick_type,
            # IB will continue sending ticks to you if set to 0
            kwargs.get("numberOfTicks", 0),
            kwargs.get("ignoreSize", False),
        )
        self._logger.debug(
            f"{self._bkr} requested (req_id={self._request_id}) {product!s} tick by tick data ({tick_type=})"
        )
        self._update_request_id_and_corresponding_product(product)

    def _request_market_depth(self, **kwargs):
        """Level 2 orderbook"""
        product = kwargs["product"]
        orderbook_depth = self._orderbook_depth[str(product)]
        self.reqMktDepth(
            self._request_id,
            product,
            orderbook_depth,
            kwargs.get("isSmartDepth", False),
            [],  # for IB internal use only
        )
        self._logger.debug(
            f"{self._bkr} requested (req_id={self._request_id}) {product!s} market depth ({orderbook_depth=})"
        )
        self._update_request_id_and_corresponding_product(product)

    def _request_real_time_bar(self, **kwargs):
        """5 Seconds Real Time Bars"""
        product = kwargs["product"]
        self.reqRealTimeBars(
            self._request_id,
            product,
            kwargs["period"],
            kwargs.get("whatToShow", "TRADES"),
            kwargs.get("useRTH", False),
            [],  # for IB internal use only
        )
        self._logger.debug(
            f"{self._bkr} requested (req_id={self._request_id}) {product!s} real time bar ({bar_size=})"
        )
        self._update_request_id_and_corresponding_product(product)

    """
    private channels
    ---------------------------------------------------
    """

    def _request_account_updates(self, account_code: str):
        """
        Args:
            account_code: e.g. U1234567
        """
        self.reqAccountUpdates(True, account)

    # TODO
    def _request_account_updates_multi(self):
        self.reqAccountUpdatesMulti()

    def _request_account_summary(self, **kwargs):
        self.reqAccountSummary(
            self._request_id,
            kwargs.get("groupName", "All"),
            kwargs.get("tags", AccountSummaryTags.AllTags),
        )
        self._increment_request_id()
