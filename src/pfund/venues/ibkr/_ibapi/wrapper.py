from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from logging import Logger
    from ibapi.contract import ContractDetails
    from pfund.venues.ibkr.api import InteractiveBrokersAPI

from abc import abstractmethod, ABC
from collections import defaultdict
from decimal import Decimal

from ibapi.wrapper import *  # noqa: F403  # pyright: ignore[reportWildcardImportFromLibrary]
from ibapi.wrapper import EWrapper, TickerId
from numpy import sign


class InteractiveBrokersWrapper(EWrapper, ABC):
    _logger: Logger

    def __init__(self):
        super().__init__()
        self._last_tick_pxs = defaultdict(dict)

    @abstractmethod
    def _on_response(self, request_id: int, data: Any) -> None: ...

    @abstractmethod
    def _on_response_end(self, request_id: int) -> None: ...

    @abstractmethod
    def _on_response_error(self, request_id: int, error: str) -> None: ...

    def error(
        self,
        reqId: TickerId,
        errorTime: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson="",
    ):
        super().error(
            reqId,
            errorTime,
            errorCode,
            errorString,
            advancedOrderRejectJson=advancedOrderRejectJson,
        )
        # informational codes, not failures of a pending request:
        # 1100-1102 = connectivity between TWS/gateway and IB servers,
        # 2100-2169 = warnings (e.g. 2104/2106/2158 "market data farm connection is OK")
        is_warning = 1100 <= errorCode <= 1102 or 2100 <= errorCode <= 2169
        if is_warning:
            return
        self._on_response_error(reqId, f"{errorCode=}: {errorString}")

    def connectAck(self):
        super().connectAck()
        self._logger.info("connected")

    def connectionClosed(self):
        super().connectionClosed()
        self._logger.warning("connection closed")

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        super().contractDetails(reqId, contractDetails)
        self._on_response(reqId, contractDetails)

    def contractDetailsEnd(self, reqId: int):
        super().contractDetailsEnd(reqId)
        self._on_response_end(reqId)

    def updateAccountTime(self, timeStamp: str):
        super().updateAccountTime(timeStamp)

    def accountDownloadEnd(self, accountName: str):
        super().accountDownloadEnd(accountName)

    def accountSummary(
        self, reqId: int, account: str, tag: str, value: str, currency: str
    ):
        super().accountSummary(reqId, account, tag, value, currency)

    def accountSummaryEnd(self, reqId: int):
        super().accountSummaryEnd(reqId)

    """
    public channels
    ---------------------------------------------------
    """

    # # TODO
    # def tickReqParams(
    #     self, tickerId: int, minTick: float, bboExchange: str, snapshotPermissions: int
    # ):
    #     super().tickReqParams(tickerId, minTick, bboExchange, snapshotPermissions)

    # # TODO
    # def tickGeneric(self, reqId: TickerId, tickType: TickType, value: float):
    #     super().tickGeneric(reqId, tickType, value)

    # def tickPrice(
    #     self, reqId: TickerId, tickType: TickType, price: float, attrib: TickAttrib
    # ):
    #     product = self._product_by_req_id[reqId]
    #     if tickType not in self._subscribed_market_data_tick_types[pdt]:
    #         return
    #     if tickType == TickTypeEnum.BID:
    #         self._last_tick_pxs[reqId][TickTypeEnum.BID_SIZE] = price
    #     elif tickType == TickTypeEnum.ASK:
    #         self._last_tick_pxs[reqId][TickTypeEnum.ASK_SIZE] = price
    #     elif tickType == TickTypeEnum.LAST:
    #         self._last_tick_pxs[reqId][TickTypeEnum.LAST_SIZE] = price
    #     super().tickPrice(reqId, tickType, price, attrib)

    # def tickSize(self, reqId: TickerId, tickType: TickType, size: Decimal):
    #     product = self._product_by_req_id[reqId]
    #     if tickType not in self._subscribed_market_data_tick_types[pdt]:
    #         return
    #     if reqId in self._last_tick_pxs and tickType in self._last_tick_pxs[reqId]:
    #         px = Decimal(self._last_tick_pxs[reqId][tickType])
    #         # When tickPrice and tickSize are reported as -1, this indicates that there is no data currently available.
    #         if px < 0 or size < 0:
    #             self._logger.warning(f"{self.bkr} {px=} or {size=} < 0")
    #             return
    #         if tickType == TickTypeEnum.BID_SIZE:
    #             bids = ((px, size),)
    #             zmq_msg = (
    #                 1,
    #                 1,
    #                 (self.bkr, product.exchange, str(product), bids, (), None),
    #             )
    #             self._zmq.send(*zmq_msg)
    #         elif tickType == TickTypeEnum.ASK_SIZE:
    #             asks = ((px, size),)
    #             zmq_msg = (
    #                 1,
    #                 1,
    #                 (self.bkr, product.exchange, str(product), (), asks, None),
    #             )
    #             self._zmq.send(*zmq_msg)
    #         elif tickType == TickTypeEnum.LAST_SIZE:
    #             zmq_msg = (
    #                 1,
    #                 2,
    #                 (self.bkr, product.exchange, str(product), px, size, None),
    #             )
    #             self._zmq.send(*zmq_msg)
    #     else:
    #         # TEMP, this is normal, but wanna log it to confirm
    #         self._logger.warning(
    #             f"{reqId=} {tickType=} {size=} cannot find matching tick price"
    #         )
    #     super().tickSize(reqId, tickType, size)

    # def tickByTickBidAsk(
    #     self,
    #     reqId: int,
    #     time: int,
    #     bidPrice: float,
    #     askPrice: float,
    #     bidSize: Decimal,
    #     askSize: Decimal,
    #     tickAttribBidAsk: TickAttribBidAsk,
    # ):
    #     product = self._product_by_req_id[reqId]
    #     bids = ((Decimal(bidPrice), bidSize),)
    #     asks = ((Decimal(askPrice), askSize),)
    #     extra = {"tickAttribBidAsk": tickAttribBidAsk}
    #     zmq_msg = (
    #         1,
    #         1,
    #         (self.bkr, product.exchange, str(product), bids, asks, time, extra),
    #     )
    #     self._zmq.send(*zmq_msg)
    #     super().tickByTickBidAsk(
    #         reqId, time, bidPrice, askPrice, bidSize, askSize, tickAttribBidAsk
    #     )

    # # TODO
    # def tickByTickMidPoint(self, reqId: int, time: int, midPoint: float):
    #     super().tickByTickMidPoint(reqId, time, midPoint)

    # def tickByTickAllLast(
    #     self,
    #     reqId: int,
    #     tickType: int,
    #     time: int,
    #     price: float,
    #     size: Decimal,
    #     tickAttribLast: TickAttribLast,
    #     exchange: str,
    #     specialConditions: str,
    # ):
    #     product = self._product_by_req_id[reqId]
    #     # TODO, HALTED
    #     # if price == 0 and size == 0 and tickAttribLast.pastLimit:

    #     extra = {
    #         "tickAttribLast": tickAttribLast,
    #         "exchange": exchange,
    #         "specialConditions": specialConditions,
    #     }
    #     zmq_msg = (
    #         1,
    #         2,
    #         (self.bkr, product.exchange, str(product), price, size, time, extra),
    #     )
    #     self._zmq.send(*zmq_msg)
    #     super().tickByTickAllLast(
    #         reqId,
    #         tickType,
    #         time,
    #         price,
    #         size,
    #         tickAttribLast,
    #         exchange,
    #         specialConditions,
    #     )

    # def updateMktDepth(
    #     self,
    #     reqId: TickerId,
    #     position: int,
    #     operation: int,
    #     side: int,
    #     price: float,
    #     size: Decimal,
    # ):
    #     self._update_orderbook(reqId, position, operation, side, price, size)
    #     super().updateMktDepth(reqId, position, operation, side, price, size)

    # def updateMktDepthL2(
    #     self,
    #     reqId: TickerId,
    #     position: int,
    #     marketMaker: str,
    #     operation: int,
    #     side: int,
    #     price: float,
    #     size: Decimal,
    #     isSmartDepth: bool,
    # ):
    #     self._update_orderbook(
    #         reqId,
    #         position,
    #         operation,
    #         side,
    #         price,
    #         size,
    #         marketMaker=marketMaker,
    #         isSmartDepth=isSmartDepth,
    #     )
    #     super().updateMktDepthL2(
    #         reqId, position, marketMaker, operation, side, price, size, isSmartDepth
    #     )

    # def realtimeBar(
    #     self,
    #     reqId: TickerId,
    #     time: int,
    #     open_: float,
    #     high: float,
    #     low: float,
    #     close: float,
    #     volume: Decimal,
    #     wap: Decimal,
    #     count: int,
    # ):
    #     resolution = "5s"  # NOTE: currently only 5 second bars are supported by IB
    #     product = self._product_by_req_id[reqId]
    #     o, h, l, c, v, ts = (
    #         Decimal(open_),
    #         Decimal(high),
    #         Decimal(low),
    #         Decimal(close),
    #         Decimal(volume),
    #         int(time),
    #     )
    #     extra = {"wap": wap, "count": count}
    #     zmq_msg = (
    #         1,
    #         3,
    #         (
    #             self.bkr,
    #             product.exchange,
    #             str(product),
    #             resolution,
    #             o,
    #             h,
    #             l,
    #             c,
    #             v,
    #             ts,
    #             extra,
    #         ),
    #     )
    #     self._zmq.send(*zmq_msg)
    #     super().realtimeBar(reqId, time, open_, high, low, close, volume, wap, count)

    # """
    # private channels
    # ---------------------------------------------------
    # """

    # # NOTE: IB gateway does not call it automatically
    # def managedAccounts(self, accountsList: str):
    #     super().managedAccounts(accountsList)
    #     # TODO for sub-accounts
    #     # account_names = accountsList.rstrip(',').split(',')

    # # EXTEND
    # def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
    #     super().updateAccountValue(key, val, currency, accountName)
    #     match key:
    #         case "TotalCashBalance" | "AvailableFunds" | "EquityWithLoanValue" if (
    #             currency != "BASE"
    #         ):
    #             ccy = self._adapter(currency)
    #             balance_type = self._adapter(key)
    #             balances = {"ts": None, "data": {ccy: {balance_type: Decimal(val)}}}
    #             zmq_msg = (
    #                 3,
    #                 1,
    #                 (
    #                     self.bkr,
    #                     "",
    #                     accountName,
    #                     balances,
    #                 ),
    #             )
    #             self._zmq.send(*zmq_msg)
    #         # TODO
    #         # case 'UnrealizedPnL' | 'RealizedPnL':
    #         #     ...
    #         # case 'InitMarginReq' | 'MaintMarginReq':
    #         #     ...
    #         # case _:
    #         #     ...

    # def updatePortfolio(
    #     self,
    #     contract: Contract,
    #     position: Decimal,
    #     marketPrice: float,
    #     marketValue: float,
    #     averageCost: float,
    #     unrealizedPNL: float,
    #     realizedPNL: float,
    #     accountName: str,
    # ):
    #     super().updatePortfolio(
    #         contract,
    #         position,
    #         marketPrice,
    #         marketValue,
    #         averageCost,
    #         unrealizedPNL,
    #         realizedPNL,
    #         accountName,
    #     )
    #     # FIXME only for stocks
    #     pdt = "_".join(
    #         [
    #             self._adapter(contract.symbol),
    #             self._adapter(contract.currency),
    #             self._adapter(contract.secType),
    #         ]
    #     )
    #     side = sign(position)
    #     # TODO, for different products, it should be adjusted
    #     positions = {
    #         "ts": None,
    #         "data": {
    #             pdt: {
    #                 side: {
    #                     "avg_px": Decimal(averageCost),
    #                     "size": position,
    #                     "market_px": Decimal(marketPrice),
    #                     "unrealized_pnl": unrealizedPNL,
    #                     "realized_pnl": realizedPNL,
    #                 }
    #             }
    #         },
    #     }
    #     zmq_msg = (
    #         3,
    #         2,
    #         (
    #             self.bkr,
    #             "",
    #             accountName,
    #             positions,
    #         ),
    #     )
    #     self._zmq.send(*zmq_msg)
