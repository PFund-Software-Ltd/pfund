from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import tEnvironment
    from pfund.datas import QuoteData, TickData, BarData

import time
try:
    import orjson as json
except ImportError:
    import json
import hmac
from decimal import Decimal

from pfund.exchanges.ws_api_base import BaseWebsocketApi
from pfund.enums import Environment, CryptoExchange, PublicDataChannel, PrivateDataChannel, DataChannelType
from pfund.products.product_bybit import BybitProduct
from pfund.datas.timeframe import TimeframeUnits


ProductCategory = BybitProduct.ProductCategory


class WebsocketApi(BaseWebsocketApi):
    name = CryptoExchange.BYBIT

    VERSION = 'v5'
    URLS = {
        Environment.PAPER: {
            'public': f'wss://stream-testnet.bybit.com/{VERSION}/public',
            'private': f'wss://stream-testnet.bybit.com/{VERSION}/private',
        },
        Environment.LIVE: {
            'public': f'wss://stream.bybit.com/{VERSION}/public',
            'private': f'wss://stream.bybit.com/{VERSION}/private'
        }
    }
    SUPPORTED_ORDERBOOK_LEVELS = {
        # category: supported orderbook levels
        ProductCategory.LINEAR: [1, 2],
        ProductCategory.INVERSE: [1, 2],
        ProductCategory.SPOT: [1, 2],
        ProductCategory.OPTION: [2],
    }
    SUPPORTED_RESOLUTIONS = {
        TimeframeUnits.QUOTE: {
            # category: orderbook depth
            ProductCategory.LINEAR: [1, 50, 200, 500],
            ProductCategory.INVERSE: [1, 50, 200, 500],
            ProductCategory.SPOT: [1, 50],
            ProductCategory.OPTION: [25, 100],
        },
        TimeframeUnits.TICK: [1],
        TimeframeUnits.MINUTE: [1, 3, 5, 15, 30, 60, 120, 240, 360, 720],
        TimeframeUnits.DAY: [1],
    }
    PUBLIC_CHANNEL_ARGS_LIMITS = {
        ProductCategory.OPTION: 2000,
        ProductCategory.SPOT: 10
    }

    def __init__(self, env: Environment | tEnvironment):
        super().__init__(env=env)
        self._channels: dict[DataChannelType, dict[ProductCategory, list[str]] | list[str]] = {
            DataChannelType.public: {
                ProductCategory.LINEAR: [],
                ProductCategory.INVERSE: [],
                ProductCategory.SPOT: [],
                ProductCategory.OPTION: [],
            },
            DataChannelType.private: [], 
        }
        # TODO: create self._servers?

    def _add_server(self, category: str):
        if category not in self._servers:
            self._servers.append(category)
            self._logger.debug(f'added server "{category}"')
            # FIXME: remove the default server
            if self.exch in self._servers:
                self._servers.remove(self.exch)
                
    def _ping(self):
        msg = {"op": "ping"}
        websockets = list(self._websockets.values())
        for ws in websockets:
            if ws and ws.sock and ws.sock.connected:
                self._send(ws, msg)

    def _authenticate(self, acc: str):
        account = self._accounts[acc]
        ws = self._websockets[acc]
        expires = int( (time.time() + 1) * 1000 )
        signature = hmac.new(
            bytes(account.secret, "utf-8"), 
            bytes(f'GET/realtime{expires}', "utf-8"), 
            digestmod="sha256"
        ).hexdigest()
        # param = f"api_key={account.key}&expires={expires}&signature={signature}"
        # private_url_extension = '?' + param
        self._logger.debug(f'ws={account.name} authenticates')
        msg = {'op': 'auth', 'args': [account.key, expires, signature]}
        self._send(ws, msg)

    def _create_ws_url(self, ws_name: str) -> str:
        if ws_name in self._servers:
            ws_url = self._urls['public'] + '/' + ws_name
        else:
            # if authenticate here,
            # set self._is_authenticating[ws_name] = True
            ws_url = self._urls['private']
        return ws_url
    
    def add_channel(
        self,
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: DataChannelType,
        data: QuoteData | TickData | BarData | None=None
    ):
        channel = super()._create_channel(channel, channel_type, data=data)
        if channel_type == DataChannelType.public:
            product: BybitProduct = data.product
            if channel not in self._channels[channel_type][product.category]:
                self._channels[channel_type][product.category].append(channel)
                self._logger.debug(f'added channel {channel}')
        else:
            if channel not in self._channels[channel_type]:
                self._channels[channel_type].append(channel)
                self._logger.debug(f'added channel {channel}')

    def _create_public_channel(self, data: QuoteData | TickData | BarData):
        channel = data.channel
        product: BybitProduct = data.product
        echannel = self._adapter(channel.value, group='channel')
        if channel == PublicDataChannel.orderbook:
            supported_orderbook_levels = self.SUPPORTED_ORDERBOOK_LEVELS[product.category]
            supported_orderbook_depths = self.SUPPORTED_RESOLUTIONS[TimeframeUnits.QUOTE][product.category]
            if data.level not in supported_orderbook_levels:
                raise NotImplementedError(f"{self.name} ({channel}.{product.symbol}) orderbook_level={data.level} is not supported, supported levels: {supported_orderbook_levels}")
            if data.depth not in supported_orderbook_depths:
                raise NotImplementedError(f"{self.name} ({channel}.{product.symbol}) orderbook_depth={data.depth} is not supported, supported depths: {supported_orderbook_depths}")
            full_channel = '.'.join([echannel, str(data.depth), product.symbol])
        elif channel == PublicDataChannel.tradebook:
            full_channel = '.'.join([echannel, product.symbol])
        elif channel == PublicDataChannel.candlestick:
            resolution, timeframe = data.resolution, data.timeframe
            if timeframe.unit not in self.SUPPORTED_RESOLUTIONS:
                raise NotImplementedError(f'{self.name} ({channel}.{product.symbol}) timeframe={str(timeframe)} is not supported, supported timeframes {list(self.SUPPORTED_RESOLUTIONS)}')
            eresolution = self._adapter(repr(resolution), group='resolution')
            full_channel = '.'.join([echannel, eresolution, product.symbol])
        else:
            raise NotImplementedError(f'{channel=} is not supported')
        return full_channel
    
    def _check_if_exceeds_public_channel_args_limits(self, ws, num_full_channels):
        is_public_channel = (ws.name in self._servers)
        if is_public_channel and ws.name in self.PUBLIC_CHANNEL_ARGS_LIMITS and num_full_channels > self.PUBLIC_CHANNEL_ARGS_LIMITS[ws.name]:
            return True
        return False

    def _subscribe(self, ws, full_channels: list[str]):
        num_full_channels = len(full_channels)
        if not self._check_if_exceeds_public_channel_args_limits(ws, num_full_channels):
            chunked_full_channels = [full_channels]
        else:
            limit = self.PUBLIC_CHANNEL_ARGS_LIMITS[ws.name]
            chunked_full_channels = [full_channels[i:i+limit] for i in range(0, num_full_channels, limit)]
        for full_channels in chunked_full_channels:
            # self._sub_num += len(full_channels)
            # number of subscription is per msg instead of per channel
            self._sub_num += 1
            msg = {'op': 'subscribe', 'args': full_channels}
            self._send(ws, msg)
            self._logger.debug(f'ws={ws.name} subscribes {full_channels}')

    def _unsubscribe(self, ws, full_channels: list[str]):
        num_full_channels = len(full_channels)
        if not self._check_if_exceeds_public_channel_args_limits(ws, num_full_channels):
            chunked_full_channels = [full_channels]
        else:
            limit = self.PUBLIC_CHANNEL_ARGS_LIMITS[ws.name]
            chunked_full_channels = [full_channels[i:i+limit] for i in range(0, num_full_channels, limit)]
        for full_channels in chunked_full_channels:
            # self._sub_num -= len(full_channels)
            # number of subscription is per msg instead of per channel
            self._sub_num -= 1
            msg = {'op': 'unsubscribe', 'args': full_channels}
            self._send(ws, msg)
            self._logger.debug(f'ws={ws.name} unsubscribes {full_channels}')

    # will receive msg=b'', ignore
    def _on_pong(self, ws, msg):
        pass

    def _process_message(self, ws, msg: dict) -> dict | None:
        ws_name = ws.name
        full_channel = msg['topic']
        if full_channel.startswith('orderbook'):
            return self._process_orderbook_l2_msg(ws_name, full_channel, msg)
        elif full_channel.startswith('publicTrade'):
            return self._process_tradebook_msg(ws_name, full_channel, msg)
        elif full_channel.startswith('kline'):
            return self._process_kline_msg(ws_name, full_channel, msg)
        # TODO, EXTEND, custom data
        # elif full_channel.startswith('tickers'):
        #     pass
        # elif full_channel.startswith('liquidation'):
        #     pass
        elif full_channel == 'position':
            return self._process_position_msg(ws_name, msg)
        elif full_channel == 'wallet':
            return self._process_balance_msg(ws_name, msg)
        elif full_channel == 'order':
            return self._process_order_msg(ws_name, msg)
        elif full_channel == 'execution':
            return self._process_trade_msg(ws_name, msg)
        else:
            self._logger.warning(f'unhandled topic ws={ws_name} msg {msg}')
    
    def _on_message(self, ws, msg: bytes):
        ws_name = ws.name
        msg = json.loads(msg)
        self._logger.debug(f'ws={ws_name} {msg=}')
        try:
            if 'op' in msg:
                op = msg['op']
                ret = msg.get('ret_msg')
                if msg.get('success', True):
                    if ret == 'pong' or op == 'pong':
                        pass
                    elif op == 'auth':
                        self._is_authenticated[ws_name] = True
                    elif op == 'subscribe':
                        self._num_subscribed += 1
                    else:
                        self._logger.warning(f'unhandled ws={ws_name} msg {msg}')
                else:
                    self._logger.error(f'ws={ws_name} unsuccessful msg {msg}')
            elif 'topic' in msg:
                if self._msg_callback is None:
                    self._process_message(ws, msg)
                else:
                    self._msg_callback(ws, msg)
            else:
                self._logger.warning(f'unhandled ws={ws_name} msg {msg}')
        except:
            self._logger.exception(f'_on_message ws={ws_name} exception {msg}:')

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
            self._is_snapshots_ready[pdt] = True
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
            # NOTE: extra_data only exists in public data, e.g. orderbook, tradebook, kline etc.
            'extra_data': {
                # 'trade_id': ('i',),
                'taker_side': ('S',),
                'px_direction': ('L',),  # e.g. PlusTick
            }
        }
        echannel, epdt = full_channel.split('.')
        pdt = self._adapter(epdt, group=ws_name)
        return super()._process_tradebook_msg(ws_name, msg, pdt, schema)
    
    def _process_kline_msg(self, ws_name, full_channel, msg):
        schema = {
            'result': 'data',
            'ts': 'ts',
            'ts_adj': 1/10**3,  # since timestamp in bybit is in mts
            'data': {
                'open': ('open', float),
                'high': ('high', float),
                'low': ('low', float),
                'close': ('close', float),
                'volume': ('volume', float),
                'ts': ('timestamp', float),
            }
        }
        echannel, eresolution, epdt = full_channel.split('.')
        resolution = self._adapter(eresolution, group='resolution')
        pdt = self._adapter(epdt, group=ws_name)
        return super()._process_kline_msg(ws_name, msg, resolution, pdt, schema)

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