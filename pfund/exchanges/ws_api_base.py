from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Literal, ClassVar, TypeAlias, Awaitable
if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection as WebSocket
    from pfund.typing import tEnvironment, ProductName, AccountName, FullDataChannel
    from pfund.adapter import Adapter
    from pfund.datas.resolution import Resolution
    from pfund.accounts.account_crypto import CryptoAccount
    from pfund.exchanges.exchange_base import BaseExchange
    from pfund.products.product_crypto import CryptoProduct

import time
import logging
import importlib
import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import reduce

try:
    import orjson as json
except ImportError:
    import json
from numpy import sign
from websockets.protocol import State

from pfund.managers.order_manager import OrderUpdateSource
from pfund.enums import Environment, Broker, CryptoExchange, PrivateDataChannel, DataChannelType


WebSocketName: TypeAlias = str


class BaseWebsocketApi(ABC):
    exch: ClassVar[CryptoExchange]
    SAMPLES_FILENAME = 'ws_api_samples.yml'

    URLS: ClassVar[dict[Environment, str | dict[Literal['public', 'private'], str]]] = {}
    SUPPORTED_ORDERBOOK_LEVELS = {}
    SUPPORTED_RESOLUTIONS = {}

    def __init__(self, env: Environment | tEnvironment):
        self._env = Environment[env.upper()]
        self._bkr = Broker.CRYPTO
        self._logger = logging.getLogger(self.exch.lower())
        Exchange: type[BaseExchange] = getattr(importlib.import_module(f'pfund.exchanges.{self.exch.lower()}.exchange'), 'Exchange')
        self._adapter: Adapter = Exchange.adapter
        self._callback: Callable[[str], Awaitable[None] | None] | None = None

        self._products: dict[ProductName, CryptoProduct] = {}
        self._accounts: dict[AccountName, CryptoAccount] = {}
        self._channels: dict[DataChannelType, list[str]] = {
            DataChannelType.public: [], 
            DataChannelType.private: []
        }
        self._websockets: dict[WebSocketName, WebSocket] = {}
    
        self._is_authenticated = defaultdict(bool)
        self._is_reconnecting = False
        self._sub_num = self._num_subscribed = 0

        # FIXME
        # self._ping_freq = 20  # in seconds
        # self._last_ping_ts = time.time()

        # self._check_connection_freq = 10
        # self._last_check_connection_ts = time.time()
        
        # self._is_snapshots_ready = defaultdict(bool)
        # self._bids_l2 = defaultdict(dict)
        # self._asks_l2 = defaultdict(dict)
        # self._last_quote_nums = defaultdict(int)
    
    @abstractmethod
    async def _subscribe(self, ws: WebSocket, channels: list[str], channel_type: DataChannelType):
        pass

    @abstractmethod
    async def _unsubscribe(self, ws: WebSocket, channels: list[str], channel_type: DataChannelType):
        pass

    @abstractmethod
    async def _authenticate(self, ws: WebSocket, account: CryptoAccount):
        pass
    
    @abstractmethod
    def _create_public_channel(self, product: CryptoProduct, resolution: Resolution):
        pass

    def _create_private_channel(self, channel: PrivateDataChannel | str):
        channel = PrivateDataChannel[channel.lower()]
        return self._adapter(channel, group='channel')
    
    def set_logger(self, logger: logging.Logger):
        self._logger = logger

    def set_callback(self, callback: Callable[[str], Awaitable[None] | None]):
        self._callback = callback

    def _get_url(self, channel_type: DataChannelType | Literal['public', 'private']) -> str:
        return self.URLS[self._env][DataChannelType[channel_type.lower()]]
    
    def _is_separate_url_for_private_channels(self) -> bool:
        public_url = self._get_url(DataChannelType.public)
        private_url = self._get_url(DataChannelType.private)
        return public_url != private_url
    
    def add_account(self, account: CryptoAccount) -> CryptoAccount:
        if account.name not in self._accounts:
            self._accounts[account.name] = account
            self._logger.debug(f'added account {account}')
        else:
            raise ValueError(f'account name {account.name} has already been added')
        return account

    def add_product(self, product: CryptoProduct) -> CryptoProduct:
        if product.name not in self._products:
            self._products[product.name] = product
            self._logger.debug(f'websocket added product {product.symbol}')
        else:
            existing_product = self._products[product.name]
            if existing_product != product:
                raise ValueError(f'product {product.symbol} has already been used for {existing_product}')
        return product
        
    def add_channel(self, channel: FullDataChannel, *, channel_type: Literal['public', 'private']):
        channel_type = DataChannelType[channel_type.lower()]
        if channel not in self._channels[channel_type]:
            self._channels[channel_type].append(channel)
            self._logger.debug(f'added {channel_type} channel {channel}')
        
    def _create_ws_name(self, account_name: str=''):
        if not account_name:
            return '_'.join([self.exch, 'ws']).lower()
        else:
            return '_'.join([account_name, 'ws']).lower()
        
    def _add_ws(self, ws_name: WebSocketName, ws: WebSocket):
        ws.name = ws_name  # HACK: assign ws_name to ws.name for conveniencec
        if ws_name in self._websockets:
            raise Exception(f'{ws_name} already exists')
        self._websockets[ws_name] = ws
        
    async def _checkup(self, num_connections: int):
        wait_secs = 5
        while len(self._websockets) != num_connections:
            self._logger.debug(f'waiting for all websockets to connect, websockets={list(self._websockets)}')
            await asyncio.sleep(1)
            wait_secs -= 1
            if wait_secs <= 0:
                raise Exception(f'{self.exch} websockets failed to connect, {num_connections=} {list(self._websockets)=}')
        # TODO: check if sub_num = num_subscribed, auth etc.
        else:
            pass
    
    # FIXME
    def _cleanup(self):
        self._websockets.clear()
        self._is_reconnecting = False
        self._sub_num = self._num_subscribed = 0
        self._is_snapshots_ready = defaultdict(bool)
        self._bids_l2 = defaultdict(dict)
        self._asks_l2 = defaultdict(dict)
        self._last_quote_nums = defaultdict(int)
        
    # FIXME
    def check_connection(self):
        if reconnect_ws_names := [ws_name for ws_name, ws in self._websockets.items() if not (self._is_connected[ws_name] and ws.sock and ws.sock.connected)]:
            self.reconnect(reconnect_ws_names)

    # FIXME
    def reconnect(self, ws_names: str|list[str]|None=None, reason: str=''):
        ws_names = self._adjust_input_ws_names(ws_names)
        if not self._is_reconnecting:
            self._logger.warning(f'{ws_names} is reconnecting, {reason=}')
            self._is_reconnecting = True
            self.disconnect(ws_names, reason='reconnection')
            self.connect(ws_names)
            self._is_reconnecting = False
        else:
            self._logger.debug(f'{ws_names} is already reconnecting, do not reconnect again due to {reason=}')
        
    async def connect(self):
        num_ws_connections = 0
        async with asyncio.TaskGroup() as task_group:
            is_empty_channels = not self._channels[DataChannelType.public] and not self._channels[DataChannelType.private]
            if is_empty_channels:
                return
            channel_type = DataChannelType.public
            url = self._get_url(channel_type)
            num_ws_connections += 1
            task_group.create_task(self._connect_ws(url))
            if self._is_separate_url_for_private_channels():
                channel_type = DataChannelType.private
                url = self._get_url(channel_type)
                for account in self._accounts.values():
                    num_ws_connections += 1
                    task_group.create_task(self._connect_ws(url, account=account))
            await self._checkup(num_ws_connections)

    async def _connect_ws(self, url: str, account: CryptoAccount | None=None):
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, ConnectionClosed

        ws_name = self._create_ws_name(account_name=account.name if account else '')
        self._logger.debug(f'{ws_name} is connecting to {url}')
        try:
            async with connect(url) as ws:
                self._add_ws(ws_name, ws)
                
                if not self._is_separate_url_for_private_channels():
                    for account in self._accounts.values():
                        self._authenticate(ws, account)
                    for channel_type, channels in self._channels.items():
                        if not channels:
                            continue
                        await self._subscribe(ws, channels, channel_type)
                else:
                    if account is None:
                        channel_type = DataChannelType.public
                    else:
                        channel_type = DataChannelType.private
                        self._authenticate(ws, account)
                    if channels := self._channels[channel_type]:
                        await self._subscribe(ws, channels, channel_type)

                try:
                    async for msg in ws:
                        await self._on_message(ws, msg)
                except ConnectionClosedOK:
                    self._logger.debug(f"{ws_name} closed normally")
                except ConnectionClosedError as e:
                    self._logger.error(f"{ws_name} closed with error: {e}")
                except ConnectionClosed as e:
                    self._logger.error(f"{ws_name} connection lost: {e}")
                except Exception:
                    self._logger.exception(f"{ws_name} unexpected error in message loop:")
        except Exception as err:
            self._logger.warning(f'{ws_name} failed to connect ({err=}), will try again later')
        
    async def disconnect(self, reason: str=''):
        for ws in self._websockets.values():
            await self._disconnect_ws(ws, reason=reason)
        self._cleanup()
            
    async def _disconnect_ws(self, ws: WebSocket, reason: str=''):
        if ws.state != State.OPEN:
            self._logger.warning(f'{ws.name} (state={ws.state.name}) is not OPEN, cannot disconnect, {reason=}')
            return
        self._logger.warning(f'{ws.name} is disconnecting, {reason=}')
        await ws.close(code=1000, reason=reason)
        await ws.wait_closed()
    
    async def _send(self, ws: WebSocket, msg: dict):
        try:
            await ws.send(json.dumps(msg))
            self._logger.debug(f'{ws.name} sent {msg}')
        except Exception:
            self._logger.exception(f'{ws.name} _send() exception:')

    def _validate_sequence_num(self, ws_name: str, pdt: str, seq_num: int, type_: Literal['quote', 'position']='quote') -> bool:
        if type_ == 'quote':
            last_seq_num = self._last_quote_nums[pdt]
        else:
            raise NotImplementedError(f'sequence number {type_=} is not supported')

        if seq_num <= last_seq_num:
            self._logger.error(f'{pdt} {type_=} {seq_num=} <= {last_seq_num=}')
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
                zmq_msg = (1, 2, (self._bkr, self.exch, pdt, tick))
                zmq.send(*zmq_msg)
        else:
            data = {'bkr': self._bkr, 'exch': self.exch, 'pdt': pdt, 'channel': 'tradebook', 'data': ticks}
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
                zmq_msg = (1, 3, (self._bkr, self.exch, pdt, bar))
                zmq.send(*zmq_msg)
        else:
            data = {'bkr': self._bkr, 'exch': self.exch, 'pdt': pdt, 'channel': f'kline.{resolution}', 'data': bars}
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
            zmq_msg = (3, 2, (self._bkr, self.exch, acc, positions))
            zmq.send(*zmq_msg)
        else:
            data = {'bkr': self._bkr, 'exch': self.exch, 'acc': acc, 'channel': 'position', 'data': positions}
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
            zmq_msg = (3, 1, (self._bkr, self.exch, acc, balances))
            zmq.send(*zmq_msg)
        else:
            data = {'bkr': self._bkr, 'exch': self.exch, 'acc': acc, 'channel': 'balance', 'data': balances}
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
            zmq_msg = (2, 1, (self._bkr, self.exch, acc, orders))
            zmq.send(*zmq_msg)
        else:
            data = {'bkr': self._bkr, 'exch': self.exch, 'acc': acc, 'channel': 'order', 'data': orders}
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
            zmq_msg = (2, 1, (self._bkr, self.exch, acc, trades))
            zmq.send(*zmq_msg)
        else:
            data = {'bkr': self._bkr, 'exch': self.exch, 'acc': acc, 'channel': 'trade', 'data': trades}
            return data
