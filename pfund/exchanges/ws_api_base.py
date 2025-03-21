from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import tCRYPTO_EXCHANGE
    from pfund.enums import Environment
    from pfund.products.product_base import BaseProduct
    from pfund.datas.data_base import BaseData

import os
import time
try:
    import orjson as json
except ImportError:
    import json
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from threading import Thread
from collections import defaultdict
from functools import reduce

from typing import Callable, Literal

from numpy import sign
from websocket import WebSocketApp, WebSocketConnectionClosedException

from pfund.managers.order_manager import OrderUpdateSource
from pfund.zeromq import ZeroMQ
from pfund.adapter import Adapter
from pfund.enums import PublicDataChannel, PrivateDataChannel, DataChannelType


class BaseWebsocketApi(ABC):
    _URLS = {}
    SUPPORTED_ORDERBOOK_LEVELS = []
    SUPPORTED_RESOLUTIONS = {}

    def __init__(self, env: Environment, name: tCRYPTO_EXCHANGE, adapter: Adapter):
        self.env = env
        self.bkr = 'CRYPTO'
        self.name = self.exch = name.upper()
        self.logger = logging.getLogger(self.exch.lower() + '_' + 'ws')
        self._adapter = adapter
        self._urls: dict|str = self.URLS.get(self.env.value, '')
        self._use_separate_private_ws_url = self._check_if_use_separate_private_ws_url()
        
        # REVIEW: is it necessary to have self.exch as a default server?
        self._servers = [self.exch]  
        
        self._full_channels = {'public': [], 'private': []}
        self._products = {}  # {pdt1: product1, pdt2: product2}
        self._accounts = {}
        self._zmqs = {}
        self._ws_threads = {}
        self._websockets = {}

        self._sub_num = self._num_subscribed = 0

        self._background_thread = None
        self._ping_timeout = 10  # in seconds
        self._ping_freq = 20  # in seconds
        self._last_ping_ts = time.time()

        self._check_connection_freq = 10
        self._last_check_connection_ts = time.time()

        self._is_connected = defaultdict(bool)
        self._is_authenticating = defaultdict(bool) 
        self._is_authenticated = defaultdict(bool)
        self._is_reconnecting = False
        
        self._orderbook_levels = {}
        self._orderbook_depths = {}
        self._is_snapshots_ready = defaultdict(bool)
        self._bids_l2 = defaultdict(dict)
        self._asks_l2 = defaultdict(dict)
        self._last_quote_nums = defaultdict(int)

        # callback function if defined by user
        self._msg_callback = None

    @property
    def URLS(self) -> dict:
        return self._URLS

    def _clean_up(self):
        self._zmqs = {}
        self._ws_threads = {}
        self._websockets = {}
        self._is_reconnecting = False
        self._sub_num = self._num_subscribed = 0
        self._is_snapshots_ready = defaultdict(bool)
        self._bids_l2 = defaultdict(dict)
        self._asks_l2 = defaultdict(dict)
        self._last_quote_nums = defaultdict(int)

    def _check_if_use_separate_private_ws_url(self):
        if type(self._urls) is dict and self._urls['public'] != self._urls['private']:
            return True
        return False

    def _run_background_tasks(self):
        while _is_running := self._websockets:
            now = time.time()
            if hasattr(self, '_ping') and now - self._last_ping_ts > self._ping_freq:
                self._ping()
                self._last_ping_ts = now
            if now - self._last_check_connection_ts > self._check_connection_freq:
                self.check_connection()
                self._last_check_connection_ts = now

    def _adjust_input_ws_names(self, ws_names: str|list[str]|None) -> list:
        if type(ws_names) is str: 
            ws_names = [ws_names]
        # case 1: ws_names is provided
        # case 2: already has some ws servers running (e.g. for disconnect())
        # case 3: no ws servers running (e.g. for connect())
        return ws_names or list(self._websockets) or self.get_all_ws_names()

    def get_servers(self):
        return self._servers

    def get_all_ws_names(self):
        return self._servers + [acc for acc in self._accounts]

    def set_msg_callback(self, callback):
        self._msg_callback = callback

    def _get_zmq(self, ws_name):
        return self._zmqs.get(ws_name, None)

    def pong(self):
        """Pongs back to Engine's ping to show that it is alive"""
        ws_name = self._servers[0]
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (4, 0, (self.bkr, self.exch, 'pong'))
            zmq.send(*zmq_msg, receiver='engine')

    def get_zmqs(self) -> list:
        return list(self._zmqs.values())

    def start_zmqs(self):
        from pfund.engines import TradeEngine
        zmq_ports = TradeEngine.settings['zmq_ports']
        for ws_name in self.get_all_ws_names():
            if ws_name in self._servers:
                if self._use_separate_private_ws_url:
                    port = zmq_ports[self.exch]['ws_api']['public'][ws_name]
                else:
                    port = zmq_ports[self.exch]['ws_api']
            elif ws_name in self._accounts:
                if self._use_separate_private_ws_url:
                    port = zmq_ports[self.exch]['ws_api']['private'][ws_name]
                else:
                    continue
            self._zmqs[ws_name] = ZeroMQ(ws_name)
            self._zmqs[ws_name].start(
                logger=self.logger,
                send_port=port,
                recv_ports=[zmq_ports['engine']]
            )
        # send the process ID to engine
        ws_name = self._servers[0]
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (4, 1, (self.bkr, self.exch, os.getpid()))
            zmq.send(*zmq_msg, receiver='engine')

    def stop_zmqs(self):
        for zmq in self._zmqs.values():
            zmq.stop()
        self._zmqs = {}

    # exchanges like bybit v5 will have one server for each category 
    # e.g. bybit has ['linear', 'inverse', 'spot', 'option']
    def add_server(self, category: str):
        if category not in self._servers:
            self._servers.append(category)
            self.logger.debug(f'added server "{category}"')
            # FIXME: remove the default server
            if self.exch in self._servers:
                self._servers.remove(self.exch)

    def add_account(self, account):
        assert account.name != self.exch, f'account name "{self.exch}" is reserved'
        self._accounts[account.name] = account
        self.logger.debug(f'added account {account.name}')

    def add_product(self, product: BaseProduct):
        if product.name in self._products:
            return
        if product.category:
            self.add_server(product.category)
        self._products[product.name] = product
        self.logger.debug(f'added product {product.name}')
    
    def remove_product(self, product: BaseProduct):
        if product.name in self._products:
            del self._products[product.name]
            self.logger.debug(f'removed product {product.name}')
        # TODO: remove server

    def add_channel(
        self,
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: DataChannelType,
        data: BaseData | None=None
    ):
        if channel in PublicDataChannel:
            self.add_product(data.product)
        full_channel = self._create_full_channel(channel, channel_type, data)
        channel_type = channel_type.value.lower()
        if full_channel not in self._full_channels[channel_type]:
            self._full_channels[channel_type].append(full_channel)
            self.logger.debug(f'added channel={full_channel}')
    
    def _create_full_channel(
        self, 
        channel: PublicDataChannel | PrivateDataChannel | str, 
        channel_type: DataChannelType, 
        data: BaseData | None=None
    ) -> str:
        if channel_type == DataChannelType.public:
            return self._create_public_channel(channel, data=data)
        elif channel_type == DataChannelType.private:
            return self._create_private_channel(channel)
    
    def remove_channel(
        self, 
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: DataChannelType,
        data: BaseData | None=None
    ):
        if channel in PublicDataChannel:
            self.remove_product(data.product)
        full_channel = self._create_full_channel(channel, channel_type, data)
        channel_type = channel_type.value.lower()
        if full_channel in self._full_channels[channel_type]:
            self._full_channels[channel_type].remove(full_channel)
            self.logger.debug(f'removed channel={full_channel}')

    # send msg to engine->connection manager to indicate it is connected 
    # to connection manager, a successful connection = connected + authenticated + subscribed + other stuff (e.g. snapshots ready)
    def _on_all_connected(self):
        ws_name = self._servers[0]
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (4, 2, (self.bkr, self.exch, 'connected'),)
            zmq.send(*zmq_msg, receiver='engine')

    def _on_all_disconnected(self):
        ws_name = self._servers[0]
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (4, 3, (self.bkr, self.exch, 'disconnected'))
            zmq.send(*zmq_msg, receiver='engine')

    def _on_connected(self, ws_name: str):
        if not self._is_connected[ws_name]:
            self._is_connected[ws_name] = True
            self.logger.debug(f'ws={ws_name} is connected')
        else:
            self.logger.warning(f'ws={ws_name} is already connected')

    def _on_disconnected(self, ws_name: str):
        if self._is_connected[ws_name]:
            self._is_connected[ws_name] = False
            self._is_authenticated[ws_name] = False
            self.logger.debug(f'ws={ws_name} is disconnected')
        else:
            self.logger.warning(f'ws={ws_name} is already disconnected')

    # connected = ws is connected
    def is_connected(self, ws_name):
        return self._is_connected[ws_name]

    def _is_all_connected(self, ws_names: list):
        return all([self._is_connected[ws_name] for ws_name in ws_names])

    def _is_all_authenticated(self, ws_names: list):
        return all([self._is_authenticated[ws_name] for ws_name in ws_names if ws_name not in self._servers])

    def _is_all_subscribed(self):
        return self._num_subscribed == self._sub_num and self._num_subscribed != 0 and self._sub_num != 0

    def _is_all_snapshots_ready(self):
        return all([self._is_snapshots_ready[pdt] for pdt in self._products])

    def _create_ws_app(self, ws_name: str, url: str) -> WebSocketApp:
        ws = WebSocketApp(
            url,
            on_open=self._on_open, 
            on_message=self._on_message,
            on_error=self._on_error, 
            on_close=self._on_close,
            on_pong=self._on_pong,
        )
        # HACK: useful self-assigned attribute to the ws object
        ws.name = ws_name
        return ws
    
    def _create_ws_thread(self, ws):
        return Thread(
            name=ws.name,
            target=ws.run_forever, 
            kwargs={
                'ping_interval': self._ping_freq,
                'ping_timeout': self._ping_timeout,
            }, 
            daemon=True
        )

    def check_connection(self):
        if reconnect_ws_names := [ws_name for ws_name, ws in self._websockets.items() if not (self._is_connected[ws_name] and ws.sock and ws.sock.connected)]:
            self.reconnect(reconnect_ws_names)

    def reconnect(self, ws_names: str|list[str]|None=None, reason: str=''):
        ws_names = self._adjust_input_ws_names(ws_names)
        if not self._is_reconnecting:
            self.logger.warning(f'{ws_names} is reconnecting, {reason=}')
            self._is_reconnecting = True
            self.disconnect(ws_names, reason='reconnection')
            self.connect(ws_names)
            self._is_reconnecting = False
        else:
            self.logger.debug(f'{ws_names} is already reconnecting, do not reconnect again due to {reason=}')
    
    def connect(self, ws_names: str|list[str]|None=None) -> bool:
        ws_names = self._adjust_input_ws_names(ws_names)
        for ws_name in ws_names:
            is_private_ws = ws_name in self._accounts
            # if no separate server for private_ws, it will share the same server with public_ws
            if is_private_ws and not self._use_separate_private_ws_url:
                continue
            ws_url = self._create_ws_url(ws_name)
            self.logger.debug(f'ws={ws_name} is connecting to {ws_url}')
            ws = self._create_ws_app(ws_name, ws_url)
            self._websockets[ws_name] = ws
            self._ws_threads[ws_name] = self._create_ws_thread(ws)
        
        # start running the ws apps
        for thd in self._ws_threads.values():
            thd.start()
            self.logger.debug(f'thread {thd.name} started')
        
        if self._wait(lambda: self._is_all_connected(ws_names), reason='connection'):
            if self._wait(lambda: self._is_all_authenticated(ws_names), reason='authentication'):
                for ws_name in ws_names:
                    if full_channels := self._full_channels['public' if ws_name in self._servers else 'private']:
                        ws = self._websockets[ws_name]
                        self._subscribe(ws, full_channels)
                if self._sub_num == 0:
                    raise Exception(f'No subscription/Empty channel for {self.exch} ws, please subscribe to at least one channel')
                if self._wait(self._is_all_subscribed, reason='subscription'):
                    if hasattr(self, '_get_initial_snapshots'):
                        Thread(target=self._get_initial_snapshots, daemon=True).start()
                        self.logger.debug(f'_get_initial_snapshots thread started')
                    pdts = list(self._orderbook_levels) if self._orderbook_levels else list(self._products)
                    # set all snapshots to be ready if orderbook is level 1
                    for pdt in pdts:
                        if self._orderbook_levels.get(pdt, 1) == 1:
                            self._is_snapshots_ready[pdt] = True
                    if self._wait(self._is_all_snapshots_ready, reason='snapshots'):
                        self._background_thread = Thread(target=self._run_background_tasks, daemon=True)
                        self._background_thread.start()
                        self.logger.debug(f'background thread started')
                        self._on_all_connected()
                        return
        self.disconnect(reason='connection failed')

    def disconnect(self, ws_names: str|list[str]|None=None, reason: str=''):
        ws_names = self._adjust_input_ws_names(ws_names)
        for ws_name in ws_names:
            ws = self._websockets[ws_name]
            self.logger.warning(f'ws={ws_name} is disconnecting, {reason=}')
            del self._websockets[ws_name]
            self.logger.warning(f'ws={ws_name} is closing')
            ws.close()
            thd = self._ws_threads[ws_name]
            while thd.is_alive():
                self.logger.debug(f'waiting for thread {thd.name} to finish')
                time.sleep(1)
            else:
                self.logger.debug(f'thread {thd.name} is finished')
                del self._ws_threads[ws_name]
        while self._background_thread and self._background_thread.is_alive():
            self.logger.debug(f'waiting for background thread to finish')
            time.sleep(1)
        else:
            self._background_thread = None
            self.logger.debug(f'background thread is finished')
        self._clean_up()
        self._on_all_disconnected()

    def _send(self, ws, msg):
        try:
            with suppress(WebSocketConnectionClosedException):
                ws.send(json.dumps(msg))
        except:
            self.logger.exception(f'ws={ws.name} exception:')

    def _on_pong(self, ws, msg):
        self.logger.warning(f'unhandled ws={ws.name} pong {msg=}')

    def _on_open(self, ws):
        self._on_connected(ws.name)
        if not self._use_separate_private_ws_url:
            for acc in self._accounts:
                self._authenticate(acc)
        else:
            # is_authenticating is True = the exchange authenticates when connecting to the ws server
            # so ws connection is opened = authenticated
            if self._is_authenticating[ws.name]:
                if ws.name not in self._servers:
                    self._is_authenticated[ws.name] = True
                    self._is_authenticating[ws.name] = False
            else:
                if ws.name in self._accounts:
                    self._authenticate(ws.name)
        self.logger.debug(f'ws={ws.name} is opened')

    @abstractmethod
    def _on_message(self, ws, msg):
        pass

    def _on_error(self, ws, error):
        self._on_disconnected(ws.name)
        self.logger.error(f'ws={ws.name} error {error}')

    def _on_close(self, ws, status_code, reason):
        self._on_disconnected(ws.name)
        self.logger.warning(f'ws={ws.name} is closed, status_code={status_code} reason={reason}')
    
    def _wait(self, condition_func: Callable, reason: str='', timeout: int=10):
        while timeout:
            if condition_func():
                self.logger.debug(f'{reason} is successful')
                return True
            timeout -= 1
            time.sleep(1)
            log_msg = f'waiting for {reason}'
            if reason in ['connection', 'disconnection']:
                log_msg += f' _is_connected: {self._is_connected}'
            elif reason == 'authentication':
                log_msg += f' _is_authenticated: {self._is_authenticated}'
            elif 'subscription' in reason:
                log_msg += f' {self._sub_num=} {self._num_subscribed=}'
            elif reason == 'snapshots':
                log_msg += f' _is_snapshots_ready: {self._is_snapshots_ready}'
            self.logger.info(log_msg)
        else:
            self.logger.warning(f'failed {log_msg}')
            return False

    def _validate_sequence_num(self, ws_name: str, pdt: str, seq_num: int, type_: Literal['quote', 'position']='quote') -> bool:
        if type_ == 'quote':
            last_seq_num = self._last_quote_nums[pdt]
        else:
            raise NotImplementedError(f'sequence number {type_=} is not supported')

        if seq_num <= last_seq_num:
            self.logger.error(f'{pdt} {type_=} {seq_num=} <= {last_seq_num=}')
            self.disconnect(ws_name, reason=f'wrong {type_}_num')
            return False
        else:           
            if type_ == 'quote':
                self._last_quote_nums[pdt] = seq_num
            return True
    
    def _process_tradebook_msg(self, ws_name, msg, pdt, schema):
        ticks = []
        res = msg[schema['result']]
        res_type = type(res)
        ts_adj = schema['ts_adj']
        msg_ts = float(step_into(msg, schema['ts'])) * ts_adj if 'ts' in schema else None
        if res_type is list:
            for trade in res:
                tick = {'ts': msg_ts, 'data': {}, 'extra_data': {}}
                for k, (ek, *sequence) in schema['data'].items():
                    initial_value = self._adapter(step_into(trade, ek))
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    tick['data'][k] = v
                    if k == 'ts':
                        tick['data'][k] *= ts_adj
                for k, (ek, *sequence) in schema.get('extra_data', {}).items():
                    initial_value = step_into(trade, ek)
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    tick['extra_data'][k] = v
                ticks.append(tick)
        else:
            raise NotImplementedError(f'{self.exch} ws trade msg {res_type=} is not supported')
        zmq = self._get_zmq(ws_name)
        if zmq:
            for tick in ticks:
                zmq_msg = (1, 2, (self.bkr, self.exch, pdt, tick))
                zmq.send(*zmq_msg)
        else:
            data = {'bkr': self.bkr, 'exch': self.exch, 'pdt': pdt, 'channel': 'tradebook', 'data': ticks}
            return data

    def _process_kline_msg(self, ws_name, msg, resolution: str, pdt, schema):
        bars = []
        res = msg[schema['result']]
        res_type = type(res)
        ts_adj = schema['ts_adj']
        msg_ts = float(step_into(msg, schema['ts'])) * ts_adj if 'ts' in schema else None
        if res_type is list:
            for kline in res:
                bar = {'ts': msg_ts, 'resolution': resolution, 'data': {}, 'extra_data': {}}
                for k, (ek, *sequence) in schema['data'].items():
                    initial_value = self._adapter(step_into(kline, ek))
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    bar['data'][k] = v
                    if k == 'ts':
                        bar['data'][k] *= ts_adj
                for k, (ek, *sequence) in schema.get('extra_data', {}).items():
                    initial_value = step_into(kline, ek)
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    bar['extra_data'][k] = v
                bars.append(bar)
        else:
            raise NotImplementedError(f'{self.exch} ws kline msg {res_type=} is not supported')
        zmq = self._get_zmq(ws_name)
        if zmq:
            for bar in bars:
                zmq_msg = (1, 3, (self.bkr, self.exch, pdt, bar))
                zmq.send(*zmq_msg)
        else:
            data = {'bkr': self.bkr, 'exch': self.exch, 'pdt': pdt, 'channel': f'kline.{resolution}', 'data': bars}
            return data

    def _process_position_msg(self, ws_name, msg, schema) -> dict:
        acc = ws_name
        positions = {'ts': None, 'data': defaultdict(dict)}
        res = step_into(msg, schema['result'])
        res_type = type(res)
        if 'ts' in schema:
            positions['ts'] = float(step_into(msg, schema['ts'])) * schema['ts_adj']
        if res_type is list:
            for position in res:
                category = step_into(position, schema['category']) if 'category' in schema else ''
                epdt = step_into(position, schema['pdt'])
                pdt = self._adapter(epdt, group=category)
                qty = float(step_into(position, schema['data']['qty'][0]))
                if qty == 0 and pdt not in self._products:
                    continue
                if 'side' in schema:
                    eside = step_into(position, schema['side'])
                    side = self._adapter(eside, group='side')
                # e.g. BINANCE_USDT only returns position size (signed qty)
                elif 'size' in schema:
                    side = sign(step_into(position, schema['size']))
                positions['data'][pdt][side] = {}
                for k, (ek, *sequence) in schema['data'].items():
                    initial_value = self._adapter(step_into(position, ek))
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    positions['data'][pdt][side][k] = v
        else:
            raise Exception(f'{self.exch} unhandled {res_type=}')
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (3, 2, (self.bkr, self.exch, acc, positions))
            zmq.send(*zmq_msg)
        else:
            data = {'bkr': self.bkr, 'exch': self.exch, 'acc': acc, 'channel': 'position', 'data': positions}
            return data

    def _process_balance_msg(self, ws_name, msg, schema):
        acc = ws_name
        balances = {'ts': None, 'data': defaultdict(dict)}
        res = step_into(msg, schema['result'])
        res_type = type(res)
        if 'ts' in schema:
            balances['ts'] = float(step_into(msg, schema['ts'])) * schema['ts_adj']
        if res_type is list:
            for balance in res:
                eccy = step_into(balance, schema['ccy'])
                ccy = self._adapter(eccy)
                for k, (ek, *sequence) in schema['data'].items():
                    initial_value = self._adapter(step_into(balance, ek))
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    balances['data'][ccy][k] = v
        else:
            raise Exception(f'{self.exch} unhandled {res_type=}')
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (3, 1, (self.bkr, self.exch, acc, balances))
            zmq.send(*zmq_msg)
        else:
            data = {'bkr': self.bkr, 'exch': self.exch, 'acc': acc, 'channel': 'balance', 'data': balances}
            return data

    def _process_order_msg(self, ws_name, msg, schema):
        acc = ws_name
        orders = {'ts': None, 'data': defaultdict(list), 'source': OrderUpdateSource.WSO}
        res = step_into(msg, schema['result'])
        res_type = type(res)
        if 'ts' in schema:
            orders['ts'] = float(step_into(msg, schema['ts'])) * schema['ts_adj']
        if res_type is list:
            for order in res:
                category = step_into(order, schema['category']) if 'category' in schema else ''
                epdt = step_into(order, schema['pdt'])
                pdt = self._adapter(epdt, group=category)
                update = {}
                for k, (ek, *sequence) in schema['data'].items():
                    group = k + 's' if k in ['tif', 'side'] else ''
                    initial_value = self._adapter(step_into(order, ek), group=group)
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    update[k] = v
                orders['data'][pdt].append(update)
        else:
            raise Exception(f'{self.exch} unhandled {res_type=}')
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (2, 1, (self.bkr, self.exch, acc, orders))
            zmq.send(*zmq_msg)
        else:
            data = {'bkr': self.bkr, 'exch': self.exch, 'acc': acc, 'channel': 'order', 'data': orders}
            return data

    def _process_trade_msg(self, ws_name, msg, schema):
        acc = ws_name
        trades = {'ts': None, 'data': defaultdict(list), 'source': OrderUpdateSource.WST}
        res = step_into(msg, schema['result'])
        res_type = type(res)
        if 'ts' in schema:
            trades['ts'] = float(step_into(msg, schema['ts'])) * schema['ts_adj']
        if res_type is list:
            for trade in res:
                category = step_into(trade, schema['category']) if 'category' in schema else ''
                epdt = step_into(trade, schema['pdt'])
                pdt = self._adapter(epdt, group=category)
                update = {}
                for k, (ek, *sequence) in schema['data'].items():
                    group = k + 's' if k in ['tif', 'side'] else ''
                    initial_value = self._adapter(step_into(trade, ek), group=group)
                    v = reduce(lambda v, f: f(v) if v else v, sequence, initial_value)
                    update[k] = v
                trades['data'][pdt].append(update)
        else:
            raise Exception(f'{self.exch} unhandled {res_type=}')
        zmq = self._get_zmq(ws_name)
        if zmq:
            zmq_msg = (2, 1, (self.bkr, self.exch, acc, trades))
            zmq.send(*zmq_msg)
        else:
            data = {'bkr': self.bkr, 'exch': self.exch, 'acc': acc, 'channel': 'trade', 'data': trades}
            return data

    @abstractmethod
    def _authenticate(self, acc: str):
        pass

    @abstractmethod
    def _create_ws_url(self, ws_name: str) -> str:
        pass
    
    @abstractmethod
    def _create_public_channel(self, channel, product, **kwargs):
        pass

    @abstractmethod
    def _create_private_channel(self, channel, **kwargs):
        pass

    @abstractmethod
    def _subscribe(self, ws, full_channels: list[str]):
        pass

    @abstractmethod
    def _unsubscribe(self, ws, full_channels: list[str]):
        pass
