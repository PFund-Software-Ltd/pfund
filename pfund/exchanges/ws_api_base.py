from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Literal, ClassVar, TypeAlias, Awaitable, Any
if TYPE_CHECKING:
    from pfund._typing import tEnvironment, ProductName, AccountName, FullDataChannel
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

from numpy import sign
from msgspec import json
from websockets.protocol import State
from websockets.asyncio.client import ClientConnection as WebSocket

from pfund.managers.order_manager import OrderUpdateSource
from pfund.enums import Environment, Broker, CryptoExchange, PrivateDataChannel, DataChannelType


WebSocketName: TypeAlias = str
Price: TypeAlias = float
class NamedWebSocket(WebSocket):
    name: str


class BaseWebSocketAPI(ABC):
    exch: ClassVar[CryptoExchange]

    URLS: ClassVar[dict[Environment, dict[DataChannelType | Literal['public', 'private'], str]]] = {}
    SUPPORTED_ORDERBOOK_LEVELS = {}
    SUPPORTED_RESOLUTIONS = {}
    CHECK_FREQ = 10  # check connections frequency (in seconds)
    PING_FREQ = 20  # application-level ping to exchange (in seconds)

    def __init__(self, env: Environment | tEnvironment):
        self._env = Environment[env.upper()]
        assert self._env != Environment.BACKTEST, f'{self._env=} is not supported in WebSocket API'
        self._bkr = Broker.CRYPTO
        self._logger = logging.getLogger(self.exch.lower())
        Exchange: type[BaseExchange] = getattr(importlib.import_module(f'pfund.exchanges.{self.exch.lower()}.exchange'), 'Exchange')
        self._adapter: Adapter = Exchange.adapter
        self._callback: Callable[[str], Awaitable[None] | None] | None = None
        self._callback_raw_msg: bool = False

        self._products: dict[ProductName, CryptoProduct] = {}
        self._accounts: dict[AccountName, CryptoAccount] = {}
        self._channels: dict[DataChannelType, list[str]] = {
            DataChannelType.public: [],
            DataChannelType.private: []
        }
    
        self._websockets: dict[WebSocketName, WebSocket] = {}
        self._sub_num = self._num_subscribed = self._ws_num = 0
        self._is_authenticated: dict[WebSocketName, bool] = defaultdict(bool)
        
        self._last_ping_ts = time.time()
        self._last_check_ts = time.time()
    
    @abstractmethod
    async def _subscribe(self, ws: NamedWebSocket, channels: list[str], channel_type: DataChannelType):
        pass

    @abstractmethod
    async def _unsubscribe(self, ws: NamedWebSocket, channels: list[str], channel_type: DataChannelType):
        pass

    @abstractmethod
    async def _authenticate(self, ws: NamedWebSocket, account: CryptoAccount):
        pass

    @abstractmethod
    async def _ping(self):
        pass
    
    @abstractmethod
    async def _on_message(self, ws_name: str, raw_msg: bytes) -> Any:
        pass
    
    @abstractmethod
    def _create_public_channel(self, product: CryptoProduct, resolution: Resolution):
        pass

    def _create_private_channel(self, channel: PrivateDataChannel | str):
        channel = PrivateDataChannel[channel.lower()]
        return self._adapter(channel, group='channel')
    
    @abstractmethod
    def _parse_message(self, msg: dict) -> dict:
        pass

    def set_logger(self, name: str):
        self._logger = logging.getLogger(name)
    
    def set_callback(self, callback: Callable[[str], Awaitable[None] | None], raw_msg: bool=False):
        '''
        Args:
            raw_msg: 
                if True, the callback will receive the raw messages.
                if False, the callback will receive parsed messages.
        '''
        self._callback = callback
        self._callback_raw_msg = raw_msg

    def _get_url(self, channel_type: DataChannelType | Literal['public', 'private']) -> str:
        if self._env == Environment.SANDBOX:
            env = Environment.LIVE
            self._logger.warning(f'{self._env} environment is using LIVE data')
        else:
            env = self._env
        return self.URLS[env][DataChannelType[channel_type.lower()]]
    
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
        channel_type: DataChannelType = DataChannelType[channel_type.lower()]
        if channel not in self._channels[channel_type]:
            self._channels[channel_type].append(channel)
            self._logger.debug(f'added {channel_type} channel {channel}')
        
    def _create_ws_name(self, account_name: str=''):
        if not account_name:
            return '_'.join([self.exch, 'ws']).lower()
        else:
            return '_'.join([account_name, 'ws']).lower()
        
    def _get_ws(self, ws_name: WebSocketName) -> NamedWebSocket:
        if not ws_name.endswith('_ws'):
            ws_name += '_ws'
        return self._websockets[ws_name]
        
    def _add_ws(self, ws_name: WebSocketName, ws: WebSocket):
        ws.name = ws_name  # HACK: assign ws_name to ws.name for conveniencec
        if ws_name in self._websockets:
            raise Exception(f'{ws_name} already exists')
        self._websockets[ws_name] = ws
        
    async def _wait(self, target_condition: Callable[[], bool], description: str, timeout: int=5):
        while not target_condition():
            self._logger.debug(description)
            await asyncio.sleep(1)
            timeout -= 1
            if timeout <= 0:
                raise Exception(f'failed {description}')
        
    async def _checkup(self):
        await self._wait(
            target_condition=lambda: len(self._websockets) == self._ws_num,
            description=f'waiting for all websockets to connect, ws_num={self._ws_num} websockets={list(self._websockets)}',
        )
        await self._wait(
            target_condition=lambda: self._num_subscribed == self._sub_num,
            description=f'waiting for all channels to be subscribed, sub_num={self._sub_num} num_subscribed={self._num_subscribed}',
        )
        if self._accounts:
            await self._wait(
                target_condition=lambda: all(self._is_authenticated[self._create_ws_name(account_name=account.name)] for account in self._accounts.values()),
                description=f'waiting for all accounts to be authenticated, is_authenticated={self._is_authenticated}',
            )
    
    def _cleanup(self):
        self._websockets.clear()
        self._sub_num = self._num_subscribed = self._ws_num = 0
        self._is_authenticated.clear()
        self._last_ping_ts = time.time()
        self._last_check_ts = time.time()

    async def _check_connection(self):
        tasks = []
        for ws_name in list(self._websockets):
            ws = self._get_ws(ws_name)
            if ws.state in [State.CLOSING, State.CLOSED]:
                await self._disconnect_ws(ws, reason='connection lost, reconnecting')
                
                # look for account used by the ws if it is a private ws
                for _account in self._accounts.values():
                    if ws_name == self._create_ws_name(account_name=_account.name):
                        channel_type = DataChannelType.private
                        account: CryptoAccount = _account
                        break
                else:
                    channel_type = DataChannelType.public
                    account = None
                
                url = self._get_url(channel_type)
                tasks.append(self._connect_ws(url, account=account))
                
        if tasks:
            await asyncio.gather(*tasks)
                
    async def _run_background_tasks(self):
        while self._websockets:
            now = time.time()
            if now - self._last_ping_ts > self.PING_FREQ:
                await self._ping()
                self._last_ping_ts = now
            if now - self._last_check_ts > self.CHECK_FREQ:
                await self._check_connection()
                self._last_check_ts = now
            await asyncio.sleep(1)
        
    async def connect(self):
        async with asyncio.TaskGroup() as task_group:
            for channel_type, channels in self._channels.items():
                if not channels:
                    continue
                url = self._get_url(channel_type)
                if channel_type == DataChannelType.public:
                    self._ws_num += 1
                    task_group.create_task(self._connect_ws(url))
                else:
                    # REVIEW: forcing pattern: one account uses one websocket connection
                    for account in self._accounts.values():
                        self._ws_num += 1
                        task_group.create_task(self._connect_ws(url, account=account))
        await self._checkup()
        await self._run_background_tasks()

    async def _connect_ws(self, url: str, account: CryptoAccount | None=None):
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, ConnectionClosed

        ws_name = self._create_ws_name(account_name=account.name if account else '')
        assert ws_name.endswith('_ws'), f'{ws_name=} must end with "_ws"'
        self._logger.debug(f'{ws_name} is connecting to {url}')
        try:
            async with connect(url) as ws:
                self._logger.debug(f'{ws_name} is connected')
                self._add_ws(ws_name, ws)
                
                if account is None:
                    channel_type = DataChannelType.public
                else:
                    channel_type = DataChannelType.private
                    await self._authenticate(ws, account)
                if channels := self._channels[channel_type]:
                    await self._subscribe(ws, channels, channel_type)

                try:
                    async for msg in ws:
                        await self._on_message(ws_name, msg)
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
        for ws_name in list(self._websockets):
            ws = self._get_ws(ws_name)
            await self._disconnect_ws(ws, reason=reason)
        self._cleanup()
            
    async def _disconnect_ws(self, ws: NamedWebSocket, reason: str=''):
        self._logger.warning(f'{ws.name} is disconnecting (state={ws.state.name}), {reason=}')
        await ws.close(code=1000, reason=reason)
        await ws.wait_closed()
        self._websockets.pop(ws.name, None)
        self._is_authenticated[ws.name] = False
        self._logger.warning(f'{ws.name} is disconnected')
    
    async def _send(self, ws: NamedWebSocket, msg: dict):
        try:
            await ws.send(json.encode(msg))
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
