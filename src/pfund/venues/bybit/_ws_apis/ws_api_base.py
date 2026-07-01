# pyright: reportUnknownLambdaType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from pfund.typing import FullDataChannel
    from pfund.venues._apis.typing import ResponseData, Schema
    from pfund.venues._apis.ws_api_base import RawMessage, WebSocketName
    from pfund.datas.resolution import Resolution
    from pfund.enums import Environment

import inspect
import os
from decimal import Decimal
from pprint import pformat

from msgspec import json

from pfund.venues._apis.ws_api_base import BaseWebSocketAPI, NamedWebSocket
from pfund.datas.timeframe import Timeframe
from pfund.venues.bybit.product import BybitProduct
from pfund.venues.bybit.account import BybitAccount
from pfund.venues.bybit.signer import BybitSigner
from pfund.enums import TradingVenue, DataChannel, DataChannelType
from pfund.venues._apis.schema_parser import SchemaParser


class BybitBaseWebSocketAPI(BaseWebSocketAPI[BybitAccount, BybitProduct]):
    venue: ClassVar[TradingVenue] = TradingVenue.BYBIT
    _signer: ClassVar[BybitSigner] = BybitSigner()

    VERSION: ClassVar[str] = "v5"
    URLS: ClassVar[dict[Environment, dict[DataChannelType, str]]] = {}

    CATEGORY: ClassVar[BybitProduct.Category]
    # it defines the maximum number of arguments allowed in the 'args' list of a WebSocket message: {'op': '...', 'args': [...]}
    PUBLIC_CHANNEL_ARGS_LIMIT: ClassVar[int] = os.sys.maxsize  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue, reportUnknownVariableType]

    @property
    def name(self) -> str:
        return f"{self.venue}_{self.CATEGORY}"

    def _unwrap_single(self, x: list[Any]) -> Any:
        """Unwrap a single-item list into its single item."""
        if len(x) != 1:
            self._logger.error(f"Expected a single item, got {len(x)}: {x}")
            return None
        else:
            return x[0]

    def _split_channels_into_batches(
        self, channels: list[str], channel_type: DataChannelType
    ) -> list[list[str]]:
        """Split channels into batches if public channel args limit is exceeded.

        For public channels, there's a limit on the number of arguments allowed in a single
        WebSocket subscription message. If this limit is exceeded, the channels are split
        into smaller batches that fit within the limit.
        """
        num_channels = len(channels)
        is_exceeded_args_limit = (channel_type == DataChannelType.public) and (
            num_channels > self.PUBLIC_CHANNEL_ARGS_LIMIT
        )
        if not is_exceeded_args_limit:
            batched_channels = [channels]
        else:
            args_limit = self.PUBLIC_CHANNEL_ARGS_LIMIT
            batched_channels = [
                channels[i : i + args_limit] for i in range(0, num_channels, args_limit)
            ]
        return batched_channels

    async def _subscribe(
        self,
        ws: NamedWebSocket,
        channels: list[FullDataChannel],
        channel_type: DataChannelType,
    ):
        batched_channels = self._split_channels_into_batches(channels, channel_type)
        for _channels in batched_channels:
            # number of subscription is per msg, not per channel
            await self._send(ws, msg={"op": "subscribe", "args": _channels})

    async def _unsubscribe(
        self,
        ws: NamedWebSocket,
        channels: list[FullDataChannel],
        channel_type: DataChannelType,
    ):
        batched_channels = self._split_channels_into_batches(channels, channel_type)
        for _channels in batched_channels:
            # number of subscription is per msg instead of per channel
            await self._send(ws, msg={"op": "unsubscribe", "args": _channels})

    async def _authenticate(self, ws: NamedWebSocket, account: BybitAccount):
        self._logger.debug(f"{ws.name} authenticates")
        msg = {"op": "auth", "args": self._signer.sign_ws_api(account)}
        await self._send(ws, msg)

    async def _ping(self):
        msg = {"op": "ping"}
        for ws in self._websockets.values():
            await self._send(ws, msg)

    async def _on_message(self, ws_name: WebSocketName, raw_msg: bytes):
        try:
            msg: dict[str, Any] = json.decode(raw_msg)
            self._logger.debug(f"{ws_name} {msg=}")

            if "op" in msg:
                op: str = msg["op"]
                ret: str | None = msg.get("ret_msg")
                if msg.get("success"):
                    if op == "auth":
                        self._is_authenticated[ws_name] = True
                    elif op == "subscribe":
                        pass
                    else:
                        self._logger.warning(f"{ws_name} unhandled msg {msg}")
                # REVIEW: check if the current ping-pong is correct
                elif ret == "pong" or op == "pong":
                    pass
                else:
                    self._logger.error(f"{ws_name} unsuccessful msg {msg}")
            elif "topic" in msg:
                if not self._callback_raw_msg:
                    msg: ResponseData = self._parse_message(ws_name, msg)
            else:
                self._logger.warning(f"{ws_name} unhandled msg {msg}")

            if self._callback:
                result = self._callback(ws_name, msg)
                if inspect.isawaitable(result):
                    await result

        except Exception:
            self._logger.exception(f"{ws_name} _on_message exception:")

    def _create_market_data_channel(
        self, product: BybitProduct, resolution: Resolution
    ):
        """Creates a full public channel name based on the product and resolution"""
        self.add_product(product)
        metadata = self.venue.venue_class.METADATA
        supported_orderbook_lvs = cast(
            dict[BybitProduct.Category, list[int]], metadata.stream_orderbook_levels
        )
        supported_orderbook_lvs = supported_orderbook_lvs[product.category]
        supported_resolutions = cast(
            dict[BybitProduct.Category, dict[Timeframe, list[int]]],
            metadata.stream_resolution_periods,
        )
        supported_resolutions = supported_resolutions[product.category]
        if resolution.is_quote():
            channel = DataChannel.orderbook
            echannel = self.adapter(channel.value, group="channel")
            orderbook_level = resolution.orderbook_level
            orderbook_depth = resolution.period
            supported_orderbook_depths = supported_resolutions[Timeframe.QUOTE]
            if orderbook_level not in supported_orderbook_lvs:
                raise NotImplementedError(
                    f"{self.venue} ({channel}.{product.symbol}) orderbook_level={orderbook_level} is not supported, supported levels: {supported_orderbook_lvs}"
                )
            if (
                orderbook_level == 1
                and orderbook_depth not in supported_orderbook_depths
            ):
                raise NotImplementedError(
                    f"{self.venue} ({channel}.{product.symbol}) orderbook_depth={orderbook_depth} is not supported, supported depths: {supported_orderbook_depths}"
                )
            full_channel = ".".join([echannel, str(orderbook_depth), product.symbol])
        elif resolution.is_tick():
            channel = DataChannel.tradebook
            echannel = self.adapter(channel.value, group="channel")
            full_channel = ".".join([echannel, product.symbol])
        elif resolution.is_bar():
            channel = DataChannel.candlestick
            echannel = self.adapter(channel.value, group="channel")
            period = resolution.period
            if resolution.timeframe not in supported_resolutions:
                raise ValueError(
                    f"{self.venue} ({channel}.{product.symbol}) {resolution=} is not supported, supported resolutions:\n{pformat(self.SUPPORTED_RESOLUTIONS)}"
                )
            elif period not in supported_resolutions[resolution.timeframe]:
                raise ValueError(
                    f"{self.venue} ({channel}.{product.symbol}) {resolution=} ({period=}) is not supported, supported periods: {self.SUPPORTED_RESOLUTIONS[resolution.timeframe]}"
                )
            eresolution = self.adapter(repr(resolution), group="resolution")
            full_channel = ".".join([echannel, eresolution, product.symbol])
        else:
            raise NotImplementedError(
                f"{resolution=} is not supported for creating public channel"
            )
        return full_channel

    def _parse_message(self, ws_name: WebSocketName, msg: RawMessage) -> ResponseData:
        channel: str = msg["topic"]
        if channel.startswith("kline"):
            return self._parse_candlestick(msg)
        elif channel.startswith("publicTrade"):
            return self._parse_tradebook(msg)
        # TODO: handle orderbook
        # elif channel.startswith('orderbook'):
        #     return BybitWebSocketAPI._parse_orderbook(msg)
        elif channel == "wallet":
            return self._parse_balance(ws_name, msg)
        elif channel == "position":
            pass
        #     return self._process_position_msg(ws_name, msg)
        elif channel == "order":
            pass
        #     return self._process_order_msg(ws_name, msg)
        elif channel == "execution":
            pass
        #     return self._process_trade_msg(ws_name, msg)
        else:
            raise NotImplementedError(f"{self.name} {channel=} is not supported")

    @staticmethod
    def _parse_candlestick(msg: RawMessage) -> ResponseData:
        schema: Schema = {
            "ts": ("ts",),
            "channel": (
                "topic",
                BybitBaseWebSocketAPI._convert_channel,
                str,  # convert DataChannel back to normal str
            ),
            "@data": ["data"],
            "data": {
                "start_ts": (
                    "start",
                    float,
                ),
                "end_ts": (
                    "end",
                    float,
                ),
                "ts": (
                    "timestamp",
                    float,
                ),
                "open": ("open", float),
                "high": ("high", float),
                "low": ("low", float),
                "close": ("close", float),
                "volume": ("volume", float),
                "is_incremental": ("confirm", lambda x: not x),
                "@extra": [],  # extra is a self-defined field, empty list = no need to parse it from anything
                "extra": {
                    "turnover": ("turnover", float),
                },
            },
        }
        data: ResponseData = SchemaParser.convert(msg, schema)
        return data

    @staticmethod
    def _parse_tradebook(msg: RawMessage) -> ResponseData:
        schema: Schema = {
            "ts": ("ts",),
            "channel": (
                "topic",
                BybitBaseWebSocketAPI._convert_channel,
                str,  # convert DataChannel back to normal str
            ),
            "@data": ["data"],
            "data": {
                "ts": (
                    "T",
                    float,
                ),
                "price": (
                    "p",
                    float,
                ),
                "volume": ("v", float, abs),
                "@extra": [],  # extra is a self-defined field, empty list = no need to parse it from anything
                "extra": {
                    "trade_id": ("i",),
                    "taker_side": ("S",),
                    "tick_direction": ("L",),
                    "is_block_trade": ("BT",),
                },
            },
        }
        data: ResponseData = SchemaParser.convert(msg, schema)
        return data

    # TODO
    @staticmethod
    def _parse_orderbook(msg: RawMessage) -> ResponseData:
        pass

    def _process_orderbook_l2_msg(self, ws_name, full_channel, msg):
        quote = {"ts": None, "data": {"bids": None, "asks": None}, "extra": {}}
        echannel, orderbook_depth, epdt = full_channel.split(".")
        pdt = self.adapter(epdt, group=ws_name)
        data = msg["data"]
        seq_num = int(data["seq"])
        msg_type = msg["type"]
        update_id = int(data["u"])
        # not 100% sure what update_id means, make sure it is a snapshot
        if update_id == 1 and msg_type != "snapshot":
            self._logger.error(
                "unexpected case: update_id=1 but it is not an orderbook snapshot"
            )
        mts = int(msg["ts"])
        quote["ts"] = mts / 10**3
        if msg_type == "snapshot":
            # bybit allows equal seq_nums, that happens when:
            # "Linear & inverse level 1 data: if 3 seconds have elapsed without a change in the orderbook, a snapshot message will be pushed again."
            # e.g.
            # {'topic': 'orderbook.1.BTCUSDT', 'type': 'snapshot', 'ts': 1682067387414, 'data': {'s': 'BTCUSDT', 'b': [['27499.90', '35.552']], 'a': [['27501.90', '0.033']], 'u': 679942, 'seq': 8064178407}}
            # after 3 seconds,
            # {'topic': 'orderbook.1.BTCUSDT', 'type': 'snapshot', 'ts': 1682067391413, 'data': {'s': 'BTCUSDT', 'b': [['27499.90', '35.552']], 'a': [['27501.90', '0.033']], 'u': 679942, 'seq': 8064178407}}
            # diff 'ts' but same 'u' and 'seq'
            if (
                self._orderbook_depths[pdt] == 1
                and seq_num == self._last_quote_nums[pdt]
            ):
                pass
            else:
                if not self._validate_sequence_num(ws_name, pdt, seq_num):
                    return
            self._bids_l2[pdt] = {}
            self._asks_l2[pdt] = {}
            bids, asks = data["b"], data["a"]
            for bid in bids:
                px, qty = bid
                self._bids_l2[pdt][px] = qty
            for ask in asks:
                px, qty = ask
                self._asks_l2[pdt][px] = qty
        elif msg_type == "delta":
            if not self._validate_sequence_num(ws_name, pdt, seq_num):
                return
            bids_l2, asks_l2 = self._bids_l2[pdt], self._asks_l2[pdt]
            bids, asks = data["b"], data["a"]
            for bid in bids:
                px, qty = bid
                # delete
                if float(qty) == 0:
                    del bids_l2[px]
                else:  # insert / update
                    bids_l2[px] = qty
            for ask in asks:
                px, qty = ask
                # delete
                if float(qty) == 0:
                    del asks_l2[px]
                else:  # insert / update
                    asks_l2[px] = qty
        depth = self._orderbook_depths[pdt]
        bids_l2 = self._bids_l2[pdt]
        asks_l2 = self._asks_l2[pdt]
        bid_pxs = sorted(bids_l2.keys(), key=lambda px: float(px), reverse=True)[:depth]
        ask_pxs = sorted(asks_l2.keys(), key=lambda px: float(px), reverse=False)[
            :depth
        ]
        quote["data"]["bids"] = tuple((px, bids_l2[px]) for px in bid_pxs)
        quote["data"]["asks"] = tuple((px, asks_l2[px]) for px in ask_pxs)
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (1, 1, (self.bkr, self.venue, pdt, quote))
            zmq.send(*zmq_msg)
        else:
            data = {
                "bkr": self.bkr,
                "exch": self.venue,
                "pdt": pdt,
                "channel": "orderbook",
                "data": quote,
            }
            return data

    def _parse_balance(self, ws_name: WebSocketName, msg: RawMessage) -> ResponseData:
        schema: Schema = {
            "ts": ("creationTime", self._convert_ms_to_seconds),
            "channel": (
                "topic",
                lambda channel: str(self.adapter(channel, group="channels")),
            ),
            "@data": ("data", self._unwrap_single),
            "data": {
                "@account": (),
                "account": {
                    "cash": ("totalWalletBalance",),
                    "equity": ("totalEquity",),
                    "available": ("totalAvailableBalance",),
                    "initial_margin": ("totalInitialMargin",),
                    "maintenance_margin": ("totalMaintenanceMargin",),
                },
                "@balances": ("coin",),
                "balances": {
                    "currency": (
                        "coin",
                        lambda asset: self.adapter(asset, group="assets"),
                    ),
                    "cash": ("walletBalance",),
                    "equity": ("equity",),
                    "locked": ("locked",),
                    "unrealized_pnl": ("unrealisedPnl",),
                },
            },
        }
        response: ResponseData = SchemaParser.convert(msg, schema)
        # use 'locked' to calculate 'available'
        data = response["data"]
        if isinstance(data, dict) and "balances" in data:
            data["balances"] = [
                {
                    **{k: v for k, v in balance.items() if k != "locked"},
                    "available": str(
                        Decimal(balance["cash"]) - Decimal(balance["locked"])
                    ),
                }
                for balance in data["balances"]
            ]
        if self._queue:
            self._emit_balance_update(ws_name, response)
        return response

    def _process_position_msg(self, ws_name, msg):
        schema = {
            "result": "data",
            "ts": "creationTime",
            "ts_adj": 1 / 10**3,
            "pdt": "symbol",
            "side": "side",
            "category": "category",
            "data": {
                "qty": ("size", str, Decimal, abs),
                "avg_px": ("entryPrice", str, Decimal),
                "liquidation_px": ("liqPrice", str, Decimal),
                "unrealized_pnl": ("unrealisedPnl", str, Decimal),
                "realized_pnl": ("cumRealisedPnl", str, Decimal),
                # 'bankruptcy_px': ('bustPrice', str, Decimal),
            },
        }
        return super()._process_position_msg(ws_name, msg, schema)

    def _process_order_msg(self, ws_name, msg):
        schema = {
            "result": "data",
            "ts": "creationTime",
            "ts_adj": 1 / 10**3,
            "pdt": "symbol",
            "category": "category",
            "data": {
                "oid": ("orderLinkId", str),
                "eoid": ("orderId", str),
                "side": ("side", int),
                "px": ("price", str, Decimal),
                "qty": ("qty", str, Decimal, abs),
                "avg_px": ("avgPrice", str, Decimal),
                "filled_qty": ("cumExecQty", str, Decimal, abs),
                # FIXME (not sure) price that triggers a stop loss/take profit order
                "trigger_px": ("triggerPrice", str, Decimal),
                "o_type": ("orderType", str),
                "status": ("orderStatus", str),
                "tif": ("timeInForce", str),
                "is_reduce_only": ("reduceOnly", bool),
            },
        }
        return super()._process_order_msg(ws_name, msg, schema)

    def _process_trade_msg(self, ws_name, msg):
        schema = {
            "result": "data",
            "ts": "creationTime",
            "ts_adj": 1 / 10**3,
            "pdt": "symbol",
            "category": "category",
            "data": {
                "oid": ("orderLinkId", str),
                "eoid": ("orderId", str),
                "side": ("side", int),
                "px": ("orderPrice", str, Decimal),
                "qty": ("orderQty", str, Decimal, abs),
                "ltp": ("execPrice", str, Decimal),
                "ltq": ("execQty", str, Decimal, abs),
                "o_type": ("orderType", str),
                "trade_ts": ("execTime", float),
                # 'trade_id': ('execId', str),
                # specific to bybit
                "trade_type": ("execType", str),
            },
        }
        return super()._process_trade_msg(ws_name, msg, schema)
