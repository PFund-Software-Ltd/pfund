from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.accounts.account_crypto import CryptoAccount

import os
import time
import hmac
import inspect
from decimal import Decimal

from msgspec import json

from pfund.parser import SchemaParser
from pfund.exchanges.ws_api_base import BaseWebSocketAPI, NamedWebSocket
from pfund.enums import Environment, CryptoExchange, PublicDataChannel, DataChannelType
from pfund.products.product_bybit import BybitProduct
from pfund.datas.timeframe import TimeframeUnits


ProductCategory = BybitProduct.ProductCategory


class BybitWebSocketAPI(BaseWebSocketAPI):
    exch = CryptoExchange.BYBIT
    CATEGORY: ClassVar[BybitProduct.ProductCategory]
    VERSION = 'v5'
    URLS = {}
    # it defines the maximum number of arguments allowed in the 'args' list of a WebSocket message: {'op': '...', 'args': [...]}
    PUBLIC_CHANNEL_ARGS_LIMIT = os.sys.maxsize
    
    def _create_ws_name(self, account_name: str=''):
        if not account_name:
            return '_'.join([self.exch, self.CATEGORY, 'ws']).lower()
        else:
            return '_'.join([account_name, 'ws']).lower()
        
    def _split_channels_into_batches(self, channels: list[str], channel_type: DataChannelType) -> list[list[str]]:
        """Split channels into batches if public channel args limit is exceeded.
        
        For public channels, there's a limit on the number of arguments allowed in a single
        WebSocket subscription message. If this limit is exceeded, the channels are split
        into smaller batches that fit within the limit.
        """
        num_channels = len(channels)
        is_exceeded_args_limit = (channel_type == DataChannelType.public) and (num_channels > self.PUBLIC_CHANNEL_ARGS_LIMIT)
        if not is_exceeded_args_limit:
            batched_channels = [channels]
        else:
            args_limit = self.PUBLIC_CHANNEL_ARGS_LIMIT
            batched_channels = [channels[i:i+args_limit] for i in range(0, num_channels, args_limit)]
        return batched_channels

    async def _subscribe(self, ws: NamedWebSocket, channels: list[str], channel_type: DataChannelType):
        batched_channels = self._split_channels_into_batches(channels, channel_type)
        for _channels in batched_channels:
            # number of subscription is per msg, not per channel
            self._sub_num += 1
            await self._send(ws, msg={'op': 'subscribe', 'args': _channels})

    async def _unsubscribe(self, ws: NamedWebSocket, channels: list[str], channel_type: DataChannelType):
        batched_channels = self._split_channels_into_batches(channels, channel_type)
        for _channels in batched_channels:
            # number of subscription is per msg instead of per channel
            self._sub_num -= 1
            await self._send(ws, msg={'op': 'unsubscribe', 'args': _channels})
    
    async def _authenticate(self, ws: NamedWebSocket, account: CryptoAccount):
        expires = int( (time.time() + 1) * 1000 )
        signature = hmac.new(
            bytes(account._secret, "utf-8"),
            bytes(f'GET/realtime{expires}', "utf-8"),
            digestmod="sha256"
        ).hexdigest()
        # param = f"api_key={account.key}&expires={expires}&signature={signature}"
        # private_url_extension = '?' + param
        self._logger.debug(f'{ws.name} authenticates')
        msg = {'op': 'auth', 'args': [account._key, expires, signature]}
        await self._send(ws, msg)
    
    async def _ping(self):
        msg = {"op": "ping"}
        for ws in self._websockets.values():
            await self._send(ws, msg)
    
    async def _on_message(self, ws_name: str, raw_msg: bytes):
        try:
            msg: dict = json.decode(raw_msg)
            self._logger.debug(f'{ws_name} {msg=}')

            if 'op' in msg:
                op: str = msg['op']
                ret: str | None = msg.get('ret_msg')
                if 'success' in msg and msg['success']:
                    if op == 'auth':
                        self._is_authenticated[ws_name] = True
                    elif op == 'subscribe':
                        self._num_subscribed += 1
                    else:
                        self._logger.warning(f'{ws_name} unhandled msg {msg}')
                # REVIEW: check if the current ping-pong is correct
                elif ret == 'pong' or op == 'pong':
                    pass
                else:
                    self._logger.error(f'{ws_name} unsuccessful msg {msg}')
            elif 'topic' in msg:
                pass
            else:
                self._logger.warning(f'{ws_name} unhandled msg {msg}')
                
            if not self._callback_raw_msg:
                msg = self._parse_message(msg)
            result = self._callback(ws_name, msg)
            if inspect.isawaitable(result):
                await result
                
        except Exception:
            self._logger.exception(f'{ws_name} _on_message exception:')
            
    def _create_public_channel(self, product: BybitProduct, resolution: Resolution):
        '''Creates a full public channel name based on the product and resolution'''
        self.add_product(product)
        if resolution.is_quote():
            channel = PublicDataChannel.orderbook
            echannel = self._adapter(channel.value, group='channel')
            orderbook_level = resolution.orderbook_level
            orderbook_depth = resolution.period
            supported_orderbook_levels = self.SUPPORTED_ORDERBOOK_LEVELS[product.category]
            supported_orderbook_depths = self.SUPPORTED_RESOLUTIONS[TimeframeUnits.QUOTE][product.category]
            if orderbook_level not in supported_orderbook_levels:
                raise NotImplementedError(f"{self.exch} ({channel}.{product.symbol}) orderbook_level={orderbook_level} is not supported, supported levels: {supported_orderbook_levels}")
            if orderbook_depth not in supported_orderbook_depths:
                raise NotImplementedError(f"{self.exch} ({channel}.{product.symbol}) orderbook_depth={orderbook_depth} is not supported, supported depths: {supported_orderbook_depths}")
            full_channel = '.'.join([echannel, str(orderbook_depth), product.symbol])
        elif resolution.is_tick():
            channel = PublicDataChannel.tradebook
            echannel = self._adapter(channel.value, group='channel')
            full_channel = '.'.join([echannel, product.symbol])
        elif resolution.is_bar():
            channel = PublicDataChannel.candlestick
            echannel = self._adapter(channel.value, group='channel')
            period, timeframe = resolution.period, resolution.timeframe
            if timeframe.unit not in self.SUPPORTED_RESOLUTIONS:
                raise ValueError(f'{self.exch} ({channel}.{product.symbol}) {resolution=} (timeframe={timeframe.unit.name}) is not supported, supported timeframes: {[tf.name for tf in self.SUPPORTED_RESOLUTIONS]}')
            elif period not in self.SUPPORTED_RESOLUTIONS[timeframe.unit]:
                raise ValueError(f'{self.exch} ({channel}.{product.symbol}) {resolution=} ({period=}) is not supported, supported periods: {self.SUPPORTED_RESOLUTIONS[timeframe.unit]}')
            eresolution = self._adapter(repr(resolution), group='resolution')
            full_channel = '.'.join([echannel, eresolution, product.symbol])
        else:
            raise NotImplementedError(f'{resolution=} is not supported for creating public channel')
        return full_channel
    
    @staticmethod
    def _parse_message(msg: dict) -> dict:
        channel: str = msg['topic']
        if channel.startswith('kline'):
            return BybitWebSocketAPI._parse_candlestick(msg)
        else:
            raise NotImplementedError(f'{BybitWebSocketAPI.exch} {channel=} is not supported')
    
    # REVIEW: schema only for linear products?
    @staticmethod
    def _parse_candlestick(msg: dict) -> dict:
        # since timestamp in bybit is in mts
        def adjust_ts(ms: int) -> float:
            return ms / 10**3
        schema = {
            'ts': ('ts', adjust_ts),
            'channel': ['topic'],
            '@data': ['data'],
            'data': {
                'open': ('open', float),
                'high': ('high', float),
                'low': ('low', float),
                'close': ('close', float),
                'volume': ('volume', float),
                'ts': ('timestamp', float, adjust_ts),
            },
            'extra_data': (
                'data',
                # add the remaining fields other than ['open', 'high', 'low', 'close', 'volume', 'timestamp'] to the extra_data
                lambda data: [
                    {k: v for k, v in bar_dict.items() if k not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']} 
                    for bar_dict in data
                ]
            ),
        }
        data: dict = SchemaParser.convert(msg, schema)
        data['data'] = data['data'][0]  # only one element in the list, access it
        data['extra_data'] = data['extra_data'][0]  # only one element in the list, access it
        data['is_incremental'] = True  # if True, it is an incremental bar update, otherwise it is a full bar update
        return data

    # FIXME: to be removed
    def _process_message(self, ws, msg: dict) -> dict | None:
        ws_name = ws.name
        channel = msg['topic']
        if channel.startswith('orderbook'):
            return self._process_orderbook_l2_msg(channel, msg)
        elif channel.startswith('publicTrade'):
            return self._process_tradebook_msg(channel, msg)
        elif channel.startswith('kline'):
            return self._parse_candlestick(msg)
        # TODO, EXTEND, custom data
        # elif channel.startswith('tickers'):
        #     pass
        # elif channel.startswith('liquidation'):
        #     pass
        elif channel == 'position':
            return self._process_position_msg(ws_name, msg)
        elif channel == 'wallet':
            return self._process_balance_msg(ws_name, msg)
        elif channel == 'order':
            return self._process_order_msg(ws_name, msg)
        elif channel == 'execution':
            return self._process_trade_msg(ws_name, msg)
        else:
            self._logger.warning(f'unhandled topic ws={ws_name} msg {msg}')

    def _process_orderbook_l2_msg(self, ws_name, full_channel, msg):
        quote = {'ts': None, 'data': {'bids': None, 'asks': None}, 'extra_data': {}}
        echannel, orderbook_depth, epdt = full_channel.split('.')
        pdt = self._adapter(epdt, group=ws_name)
        data = msg['data']
        seq_num = int(data['seq'])
        msg_type = msg['type']
        update_id = int(data['u'])
        # not 100% sure what update_id means, make sure it is a snapshot
        if update_id == 1 and msg_type != 'snapshot':
            self._logger.error('unexpected case: update_id=1 but it is not an orderbook snapshot')
        mts = int(msg['ts'])
        quote['ts'] = mts / 10**3
        if msg_type == 'snapshot':
            # bybit allows equal seq_nums, that happens when:
            # "Linear & inverse level 1 data: if 3 seconds have elapsed without a change in the orderbook, a snapshot message will be pushed again."
            # e.g. 
            # {'topic': 'orderbook.1.BTCUSDT', 'type': 'snapshot', 'ts': 1682067387414, 'data': {'s': 'BTCUSDT', 'b': [['27499.90', '35.552']], 'a': [['27501.90', '0.033']], 'u': 679942, 'seq': 8064178407}}
            # after 3 seconds,
            # {'topic': 'orderbook.1.BTCUSDT', 'type': 'snapshot', 'ts': 1682067391413, 'data': {'s': 'BTCUSDT', 'b': [['27499.90', '35.552']], 'a': [['27501.90', '0.033']], 'u': 679942, 'seq': 8064178407}}
            # diff 'ts' but same 'u' and 'seq'
            if self._orderbook_depths[pdt] == 1 and seq_num == self._last_quote_nums[pdt]:
                pass
            else:
                if not self._validate_sequence_num(ws_name, pdt, seq_num):
                    return
            self._bids_l2[pdt] = {}
            self._asks_l2[pdt] = {}
            bids, asks = data['b'], data['a']
            for bid in bids:
                px, qty = bid
                self._bids_l2[pdt][px] = qty
            for ask in asks:
                px, qty = ask
                self._asks_l2[pdt][px] = qty
        elif msg_type == 'delta':
            if not self._validate_sequence_num(ws_name, pdt, seq_num):
                return
            bids_l2, asks_l2 = self._bids_l2[pdt], self._asks_l2[pdt]
            bids, asks = data['b'], data['a']
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
        ask_pxs = sorted(asks_l2.keys(), key=lambda px: float(px), reverse=False)[:depth]
        quote['data']['bids'] = tuple((px, bids_l2[px]) for px in bid_pxs)
        quote['data']['asks'] = tuple((px, asks_l2[px]) for px in ask_pxs)
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (1, 1, (self.bkr, self.exch, pdt, quote))
            zmq.send(*zmq_msg)
        else:
            data = {'bkr': self.bkr, 'exch': self.exch, 'pdt': pdt, 'channel': 'orderbook', 'data': quote}
            return data

    def _process_tradebook_msg(self, ws_name, full_channel, msg):
        schema = {
            'result': 'data',
            'ts': 'ts',
            'ts_adj': 1/10**3,  # since timestamp in bybit is in mts
            'data': {
                'px': ('p', float,),
                'qty': ('v', float, abs),
                'ts': ('T', float),
            },
            # NOTE: extra_data only exists in public data, e.g. orderbook, tradebook, candlestick etc.
            'extra_data': {
                # 'trade_id': ('i',),
                'taker_side': ('S',),
                'px_direction': ('L',),  # e.g. PlusTick
            }
        }
        echannel, epdt = full_channel.split('.')
        pdt = self._adapter(epdt, group=ws_name)
        return super()._process_tradebook_msg(ws_name, msg, pdt, schema)

    def _process_position_msg(self, ws_name, msg):
        schema = {
            'result': 'data',
            'ts': 'creationTime',
            'ts_adj': 1/10**3,
            'pdt': 'symbol',
            'side': 'side',
            'category': 'category',
            'data': {
                'qty': ('size', str, Decimal, abs),
                'avg_px': ('entryPrice', str, Decimal),
                'liquidation_px': ('liqPrice', str, Decimal),
                'unrealized_pnl': ('unrealisedPnl', str, Decimal),
                'realized_pnl': ('cumRealisedPnl', str, Decimal),
                # 'bankruptcy_px': ('bustPrice', str, Decimal),
            },
        }
        return super()._process_position_msg(ws_name, msg, schema)

    def _process_balance_msg(self, ws_name, msg):
        schema = {
            'result': ['data', 0, 'coin'],  # HACK
            'ts': 'creationTime',
            'ts_adj': 1/10**3,
            'ccy': 'coin',
            'data': {
                'wallet': ('walletBalance', str, Decimal),
                'available': ('availableToWithdraw', str, Decimal),
                'margin': ('equity', str, Decimal),
            },
        }
        # NOTE: need to make sure msg['data'] has only one element so that the HACK ['data', 0, 'coin'] above can work

        assert len(msg['data']) == 1
        return super()._process_balance_msg(ws_name, msg, schema)
    
    def _process_order_msg(self, ws_name, msg):
        schema = {
            'result': 'data',
            'ts': 'creationTime',
            'ts_adj': 1/10**3,
            'pdt': 'symbol',
            'category': 'category',
            'data': {
                'oid': ('orderLinkId', str),
                'eoid': ('orderId', str),
                'side': ('side', int),
                'px': ('price', str, Decimal),
                'qty': ('qty', str, Decimal, abs),
                'avg_px': ('avgPrice', str, Decimal),
                'filled_qty': ('cumExecQty', str, Decimal, abs),
                # FIXME (not sure) price that triggers a stop loss/take profit order
                'trigger_px': ('triggerPrice', str, Decimal),
                'o_type': ('orderType', str),
                'status': ('orderStatus', str),
                'tif': ('timeInForce', str),
                'is_reduce_only': ('reduceOnly', bool),
            },
        }
        return super()._process_order_msg(ws_name, msg, schema)

    def _process_trade_msg(self, ws_name, msg):
        schema = {
            'result': 'data',
            'ts': 'creationTime',
            'ts_adj': 1/10**3,
            'pdt': 'symbol',
            'category': 'category',
            'data': {
                'oid': ('orderLinkId', str),
                'eoid': ('orderId', str),
                'side': ('side', int),
                'px': ('orderPrice', str, Decimal),
                'qty': ('orderQty', str, Decimal, abs),
                'ltp': ('execPrice', str, Decimal),
                'ltq': ('execQty', str, Decimal, abs),
                'o_type': ('orderType', str),
                'trade_ts': ('execTime', float),
                # 'trade_id': ('execId', str),
                
                # specific to bybit
                'trade_type': ('execType', str),
            }
        }
        return super()._process_trade_msg(ws_name, msg, schema)