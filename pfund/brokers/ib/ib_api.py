"""This is mainly a wrapper class of IB's official API.
Conceptually, this is equivalent to ws_api_base.py in crypto
"""
import time
import os
import logging
from collections import defaultdict

from typing import Callable

from pfund.brokers.ib.ib_client import IBClient
from pfund.brokers.ib.ib_wrapper import *
from pfund.enums import PublicDataChannel, PrivateDataChannel
from pfund.datas.timeframe import TimeframeUnits


class IBApi(IBClient, IBWrapper):
    DEFAULT_ORDERBOOK_LEVEL = 2
    DEFAULT_ORDERBOOK_DEPTH = 5
    SUPPORTED_ORDERBOOK_LEVELS = [1, 2]
    SUPPORTED_RESOLUTIONS = {
        TimeframeUnits.TICK: [1],
        TimeframeUnits.SECOND: [5],
    }
    # EXTEND
    PTYPES_WITHOUT_TICK_BY_TICK_DATA = ['OPT']
    # product types which cannot subscribe to 'Last'/'AllLast' in reqTickByTickData()
    PTYPES_WITHOUT_TICK_BY_TICK_LAST_DATA = ['OPT', 'FX']

    def __init__(self, env, adapter):
        IBClient.__init__(self)
        IBWrapper.__init__(self)
        self.env = env.upper()
        self.name = self.bkr = 'IB'
        self.logger = logging.getLogger(self.name.lower())
        self._adapter = adapter
        self._full_channels = {'public': [], 'private': []}
        self._ib_params_for_channels_subscription = {}
        self._products = {}  # {pdt1: product1, pdt2: product2}
        self.account = None
        self._bids = defaultdict(list)
        self._asks = defaultdict(list)
        self._zmq = None
        self._sub_num = 0
        self._num_subscribed = 0
        self._is_connected = False
        self._is_reconnecting = False
        self._background_task_freq = 10  # in seconds
        self._background_thread = None
        self._orderbook_level = {}
        self._orderbook_depth = {}
        # since reqMktData will subscribe to bid/ask + last price/quantity automatically,
        # use this to save down which tick types the system has subscribed
        self._subscribed_market_data_tick_types = defaultdict(list)
        self._ib_thread = None
    

    """
    PFund's functions for controlling the API's connectivity,
    just like ws_api_base.py for crypto exchanges
    ---------------------------------------------------
    """
    def reconnect(self):
        if not self._is_reconnecting:
            self.logger.warning(f'{self.bkr} is reconnecting')
            self._is_reconnecting = True
            self.disconnect()
            self.connect()
            self._is_reconnecting = False
        else:
            self.logger.debug(f'{self.bkr} is already reconnecting, do not reconnect again')
    
    def add_account(self, account):
        self.account = account
        self.logger.debug(f'added account {account.name}')
            
    def add_product(self, product, **kwargs):
        self._products[str(product)] = product
        self.logger.debug(f'added product {str(product)}')
            
    def add_channel(self, channel, type_, product=None, account=None, **kwargs):
        if type_ == 'public':
            full_channel = self._create_public_channel(channel, product, **kwargs)
        elif type_ == 'private':
            full_channel = self._create_private_channel(channel, account, **kwargs)
        if full_channel not in self._full_channels[type_]:
            self._full_channels[type_].append(full_channel)
            self.logger.debug(f'added {full_channel=}')

    def is_connected(self):
        return self._is_connected
    
    def _on_connected(self):
        if not self._is_connected:
            self._is_connected = True
            zmq_msg = (4, 2, (self.bkr, '', 'connected'),)
            self._zmq.send(*zmq_msg, receiver='engine')
            self.logger.debug(f'{self.bkr} is connected')
        else:
            self.logger.warning(f'{self.bkr} is already connected')

    def _on_disconnected(self):
        if self._is_connected:
            self._is_connected = False
            zmq_msg = (4, 3, (self.bkr, '', 'disconnected'))
            self._zmq.send(*zmq_msg, receiver='engine')
            self.logger.debug(f'{self.bkr} is disconnected')
        else:
            self.logger.warning(f'{self.bkr} is already disconnected')

    def _create_public_channel(self, channel: PublicDataChannel, product, **kwargs):
        """Creates publich channel for internal use.
        Since IB's subscription does not require channel name,
        this function creates channel only for internal use, clarity and consistency.
        """
        pdt = str(product)
        epdt = self._adapter(pdt)
        echannel = self._adapter(channel)
        if channel in PublicDataChannel:
            if channel == PublicDataChannel.orderbook:
                full_channel = '.'.join([channel, pdt])
                self._orderbook_level[pdt] = int(kwargs.get('orderbook_level', self.DEFAULT_ORDERBOOK_LEVEL))
                if self._orderbook_level[pdt] not in self.SUPPORTED_ORDERBOOK_LEVELS:
                    raise NotImplementedError(f'{pdt} orderbook_level={self._orderbook_level[pdt]} is not supported')
                if 'orderbook_depth' in kwargs:
                    self._orderbook_depth[pdt] = int(kwargs['orderbook_depth'])
                elif 'num_rows' in kwargs:  # `num_rows` is a params in IB's reqMktDepth(...)
                    self._orderbook_depth[pdt] = int(kwargs['num_rows'])
                else:
                    self._orderbook_depth[pdt] = self.DEFAULT_ORDERBOOK_DEPTH
            elif channel == PublicDataChannel.tradebook:
                full_channel = '.'.join([echannel, epdt])
            elif channel == PublicDataChannel.candlestick:
                period, timeframe = kwargs['period'], kwargs['timeframe']
                if timeframe not in self.SUPPORTED_RESOLUTIONS.keys():
                    raise NotImplementedError(f'({channel}.{pdt}) {timeframe=} for kline is not supported, only timeframes in {list(self.SUPPORTED_RESOLUTIONS)} are supported')
                resolution = str(period) + timeframe
                full_channel = '.'.join([echannel, epdt, resolution])
            else:
                raise NotImplementedError(f'{channel=} is not supported')
        else:
            full_channel = channel
        kwargs.update({'product': product})
        self._ib_params_for_channels_subscription[full_channel] = kwargs
        return full_channel

    def _create_private_channel(self, channel: PrivateDataChannel, **kwargs):
        echannel = self._adapter(channel)
        account = kwargs['account']
        full_channel = '.'.join([echannel, account.name])
        self._ib_params_for_channels_subscription[full_channel] = kwargs
        return full_channel

    def _wait(self, condition_func: Callable, reason: str='', timeout: int=10):
        while timeout:
            if condition_func():
                self.logger.debug(f'{reason} is successful')
                return True
            timeout -= 1
            time.sleep(1)
            self.logger.debug(f'waiting for {reason}')
        else:
            self.logger.error(f'failed waiting for {reason}')
            return False
        
    def _subscribe(self):
        # subscribe to public channels
        for type_, full_channels in self._full_channels.items():
            self._sub_num += len(full_channels)
            if not full_channels:
                continue
            for full_channel in full_channels:
                ib_params = self._ib_params_for_channels_subscription[full_channel]
                if type_ == 'public':
                    channel, pdt, *_ = full_channel.split('.')
                    product = self._products[pdt]
                    if channel == 'orderbook':
                        if self._orderbook_level[pdt] == 1:
                            if product.ptype not in self.PTYPES_WITHOUT_TICK_BY_TICK_DATA:
                                tick_type = ib_params.get('tickType', 'BidAsk')
                                assert tick_type in ['MidPoint', 'BidAsk'], f'tickType={tick_type} is not supported for trade channel'
                                self._request_tick_by_tick_data(tick_type, **ib_params)
                            else:
                                self._request_market_data(**ib_params)
                                self._subscribed_market_data_tick_types[pdt].extend([TickTypeEnum.BID, TickTypeEnum.BID_SIZE, TickTypeEnum.ASK, TickTypeEnum.ASK_SIZE])
                        elif self._orderbook_level[pdt] == 2:
                            self._request_market_depth(**ib_params)
                    elif channel == 'tradebook':
                        if product.ptype not in self.PTYPES_WITHOUT_TICK_BY_TICK_DATA + self.PTYPES_WITHOUT_TICK_BY_TICK_LAST_DATA:
                            tick_type = ib_params.get('tickType', 'Last')
                            assert tick_type in ['Last', 'AllLast'], f'tickType={tick_type} is not supported for trade channel'
                            self._request_tick_by_tick_data(tick_type, **ib_params)
                        else:
                            self._request_market_data(**ib_params)
                            self._subscribed_market_data_tick_types[pdt].extend([TickTypeEnum.LAST, TickTypeEnum.LAST_SIZE])
                    elif channel == 'kline':
                        self._request_real_time_bar(**ib_params)
                    
                    # if did not request market data but defined related params for it, request for it anyways
                    if not self._subscribed_market_data_tick_types[pdt] and \
                        any(params in ib_params for params in ['genericTickList', 'snapshot', 'regulatorySnapshot']):
                        self._request_market_data(**ib_params)
                else:
                    channel, acc = full_channel.split('.')
                    if channel == 'account_update':
                        self._request_account_updates(acc)
                    elif channel == 'account_summary':
                        self._request_account_summary(**ib_params)

            self.logger.debug(f'{self.bkr} subscribes {full_channels}')
    
    def _unsubscribe(self):
        self._sub_num = 0
        self._num_subscribed = 0
    
    def _check_connection(self):
        if reconnect_ws_names := [ws_name for ws_name, ws in self._websockets.items() if not (self._is_connected[ws_name] and ws.sock and ws.sock.connected)]:
            self.reconnect()

    def _run_background_tasks(self):
        while _is_running:=self._websockets:
            self._check_connection()
            time.sleep(self._background_task_freq)

    def _is_all_subscribed(self):
        return (self._num_subscribed == self._sub_num and self._num_subscribed != 0 and self._sub_num != 0)

    def _update_orderbook(self, req_id, position: int, operation: int, side: int, px, qty, **kwargs):
        '''
        Args:
            position: the orderbook's row being updated
            operation: 0 = insert, 1 = update, 2 = remove
            side: 0 = ask, 1 = bid
        '''
        # boa = bids or asks
        def _update(boa: list):
            if operation == 0:
                boa.insert(position, (Decimal(px), qty))
            elif operation == 1:
                boa[position] = (Decimal(px), qty)
            elif operation == 2:
                del boa[position]
        try:
            pdt = self._req_id_to_product[req_id]
            if side == 0:
                bids, asks = None, self._asks[pdt]
                _update(asks)
            else:
                bids, asks = self._bids[pdt], None
                _update(bids)
            zmq_msg = (1, 1, (self.bkr, product.exch, str(product), bids, asks, None, kwargs))
            self._zmq.send(*zmq_msg, receiver='engine')
        except:
            self.logger.exception(f'_update_orderbook exception ({position=} {operation=} {side=} {px=} {qty=} {kwargs=}):')

    def pong(self):
        """Pongs back to Engine's ping to show that it is alive"""
        zmq_msg = (4, 0, (self.bkr, '', 'pong'))
        self._zmq.send(*zmq_msg, receiver='engine')

    def get_zmqs(self) -> list:
        return [self._zmq]

    def start_zmqs(self):
        from pfund.engines.trade_engine import TradeEngine
        zmq_ports = TradeEngine.settings['zmq_ports']
        self._zmq = ZeroMQ(self.name)
        self._zmq.start(
            logger=self.logger,
            send_port=zmq_ports[self.name],
            recv_ports=[zmq_ports['engine']]
        )
        # send the process ID to engine
        zmq_msg = (4, 1, (self.bkr, '', os.getpid(),))
        self._zmq.send(*zmq_msg, receiver='engine')

    def stop_zmqs(self):
        self._zmq.stop()
        self._zmq = None