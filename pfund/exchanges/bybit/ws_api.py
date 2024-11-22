import time
from pathlib import Path
try:
    import orjson as json
except ImportError:
    import json
import hmac
from decimal import Decimal

from pfund.const.enums import PublicDataChannel, PrivateDataChannel
from pfund.exchanges.ws_api_base import BaseWebsocketApi
from pfund.const.enums import Environment


class WebsocketApi(BaseWebsocketApi):
    DEFAULT_ORDERBOOK_DEPTH = 1
    _URLS = {
        'PAPER': {
            'public': 'wss://stream-testnet.bybit.com/v5/public',
            'private': 'wss://stream-testnet.bybit.com/v5/private',
        },
        'LIVE': {
            'public': 'wss://stream.bybit.com/v5/public',
            'private': 'wss://stream.bybit.com/v5/private'
        }
    }
    SUPPORTED_ORDERBOOK_LEVELS = [2]
    SUPPORTED_TIMEFRAMES_AND_PERIODS = {
        # normal definition of 'q' if not per category:
        # 'q': [1, 50, 200, 500],  # quote

        # per category
        'q': {
            'linear': [1, 50, 200, 500],
            'inverse': [1, 50, 200, 500],
            'spot': [1, 50],
            'option': [25, 100],
        },
        't': [1],  # tick
        'm': [1, 3, 5, 15, 30, 60, 120, 240, 360, 720],  # minute
        'd': [1],  # day
        'w': [1],  # week
        'M': [1],  # month
    }
    PUBLIC_CHANNEL_ARGS_LIMITS = {
        'option': 2000,
        'spot': 10
    }

    def __init__(self, env: Environment, adapter):
        exch = Path(__file__).parent.name
        super().__init__(env, exch, adapter)

    @property
    def URLS(self) -> dict:
        return self._URLS

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
        self.logger.debug(f'ws={account.name} authenticates')
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

    def _create_public_channel(self, channel: PublicDataChannel, product, **kwargs):
        pdt = product.pdt
        epdt = self._adapter(pdt, ref_key=product.category)
        echannel = self._adapter(channel)
        if channel in PublicDataChannel:
            if channel == PublicDataChannel.ORDERBOOK:
                self._orderbook_levels[pdt] = int(kwargs.get('orderbook_level', self.DEFAULT_ORDERBOOK_LEVEL))
                if self._orderbook_levels[pdt] not in self.SUPPORTED_ORDERBOOK_LEVELS:
                    raise NotImplementedError(f'{pdt} orderbook_level={self._orderbook_levels[pdt]} is not supported')
                self._orderbook_depths[pdt] = int(kwargs.get('orderbook_depth', self.DEFAULT_ORDERBOOK_DEPTH))
                if product.category:
                    supported_periods = self.SUPPORTED_TIMEFRAMES_AND_PERIODS['q'][product.category]
                else:
                    supported_periods = self.SUPPORTED_TIMEFRAMES_AND_PERIODS['q']
                if self._orderbook_depths[pdt] not in supported_periods:
                    # Find an integer in self.SUPPORTED_TIMEFRAMES_AND_PERIODS['q'] that is the nearest to the intended orderbook_depth
                    nearest_depth = min(supported_periods, key=lambda supported_period: abs(supported_period - self._orderbook_depths[pdt]))
                    self.logger.warning(f'orderbook_depth={self._orderbook_depths[pdt]} is not supported, using the nearest supported depth "{nearest_depth}". {supported_periods=}')
                    subscribed_orderbook_depth = nearest_depth
                else:
                    subscribed_orderbook_depth = self._orderbook_depths[pdt]
                self._orderbook_depths[pdt] = min(self._orderbook_depths[pdt], subscribed_orderbook_depth)
                full_channel = '.'.join([echannel, str(subscribed_orderbook_depth), epdt])
            elif channel == PublicDataChannel.TRADEBOOK:
                full_channel = '.'.join([echannel, epdt])
            elif channel == PublicDataChannel.KLINE:
                period, timeframe = kwargs['period'], kwargs['timeframe']
                if timeframe not in self.SUPPORTED_TIMEFRAMES_AND_PERIODS.keys():
                    raise NotImplementedError(f'({channel}.{pdt}) {timeframe=} for kline is not supported, only timeframes in {list(self.SUPPORTED_TIMEFRAMES_AND_PERIODS)} are supported')
                resolution = str(period) + timeframe
                eresolution = self._adapter(resolution)
                full_channel = '.'.join([echannel, eresolution, epdt])
            else:
                raise NotImplementedError(f'{channel=} is not supported')
        else:  # channels like 'tickers.{symbol}', 'liquidation.{symbol}'
            full_channel = '.'.join([echannel, epdt])
        return full_channel

    def _create_private_channel(self, channel: PrivateDataChannel, **kwargs):
        echannel = self._adapter(channel)
        full_channel = echannel
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
            self.logger.debug(f'ws={ws.name} subscribes {full_channels}')

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
            self.logger.debug(f'ws={ws.name} unsubscribes {full_channels}')

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
            self.logger.warning(f'unhandled topic ws={ws_name} msg {msg}')
    
    def _on_message(self, ws, msg: bytes):
        ws_name = ws.name
        msg = json.loads(msg)
        self.logger.debug(f'ws={ws_name} {msg=}')
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
                        self.logger.warning(f'unhandled ws={ws_name} msg {msg}')
                else:
                    self.logger.error(f'ws={ws_name} unsuccessful msg {msg}')
            elif 'topic' in msg:
                if self._msg_callback is None:
                    self._process_message(ws, msg)
                else:
                    self._msg_callback(ws, msg)
            else:
                self.logger.warning(f'unhandled ws={ws_name} msg {msg}')
        except:
            self.logger.exception(f'_on_message ws={ws_name} exception {msg}:')

    def _process_orderbook_l2_msg(self, ws_name, full_channel, msg):
        quote = {'ts': None, 'data': {'bids': None, 'asks': None}, 'other_info': {}}
        echannel, orderbook_depth, epdt = full_channel.split('.')
        pdt = self._adapter(epdt, ref_key=ws_name)
        data = msg['data']
        seq_num = int(data['seq'])
        msg_type = msg['type']
        update_id = int(data['u'])
        # not 100% sure what update_id means, make sure it is a snapshot
        if update_id == 1 and msg_type != 'snapshot':
            self.logger.error('unexpected case: update_id=1 but it is not an orderbook snapshot')
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
            # NOTE: other_info only exists in public data, e.g. orderbook, tradebook, kline etc.
            'other_info': {
                # 'trade_id': ('i',),
                'taker_side': ('S',),
                'px_direction': ('L',),  # e.g. PlusTick
            }
        }
        echannel, epdt = full_channel.split('.')
        pdt = self._adapter(epdt, ref_key=ws_name)
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
            # TODO: (move to OKX ws_api):
            # 'ts_adj': 1/10**3,
            # 'data': {
            #     'open': (1, float),
            #     'high': (2, float),
            #     'low': (3, float),
            #     'close': (4, float),
            #     'volume': (5, float),
            #     'ts': (0, float),
            # }
        }
        echannel, eresolution, epdt = full_channel.split('.')
        resolution = self._adapter(eresolution)
        pdt = self._adapter(epdt, ref_key=ws_name)
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