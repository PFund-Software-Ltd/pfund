"""This is mainly a wrapper class of IB's official API.
Conceptually, this is equivalent to ws_api_base.py in crypto
"""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Awaitable, Literal
if TYPE_CHECKING:
    from pfund._typing import tEnvironment, ProductName, AccountName, FullDataChannel
    from pfund.accounts.account_ib import IBAccount
    from pfund.enums import Environment
    from pfund.datas.resolution import Resolution
    from pfund.products.product_ib import IBProduct
    
import logging
from collections import defaultdict

from pfund.brokers.ib.ib_client import IBClient
from pfund.brokers.ib.ib_wrapper import IBWrapper
from pfund.brokers.ib.ib_wrapper import *
from pfund.datas.timeframe import TimeframeUnits
from pfund.enums import Environment, Broker, PublicDataChannel, PrivateDataChannel, TraditionalAssetType, DataChannelType


# this is similar to ws_api_base.py for crypto exchanges
class IBAPI(IBClient, IBWrapper):
    SUPPORTED_ORDERBOOK_LEVELS = [1, 2]
    SUPPORTED_RESOLUTIONS = {
        TimeframeUnits.TICK: [1],
        TimeframeUnits.SECOND: [5],
    }
    CHECK_FREQ = 10  # check connections frequency (in seconds)
    PING_FREQ = 20  # application-level ping to exchange (in seconds)
    
    ASSET_TYPES_WITHOUT_TICK_BY_TICK_DATA = [
        TraditionalAssetType.OPT,
        # EXTEND
        # TraditionalAssetType.INDEX,
        # TraditionalAssetType.COMBO
    ]
    # asset types that cannot subscribe to 'Last'/'AllLast' in reqTickByTickData()
    ASSET_TYPES_WITHOUT_TICK_BY_TICK_LAST_DATA = [
        TraditionalAssetType.OPT, 
        TraditionalAssetType.FX,
    ]

    def __init__(self, env: Environment | tEnvironment):
        from pfund.brokers.ib.broker_ib import IBBroker
        
        IBClient.__init__(self)
        IBWrapper.__init__(self)
        self._env = Environment[env.upper()]
        self._bkr = Broker.IB
        self._logger = logging.getLogger(self._bkr.lower())
        self._adapter = IBBroker.adapter
        self._callback: Callable[[str], Awaitable[None] | None] | None = None
        self._callback_raw_msg: bool = False

        self._products: dict[ProductName, IBProduct] = {}
        self._accounts: dict[AccountName, IBAccount] = {}
        self._channels: dict[DataChannelType, list[str]] = {
            DataChannelType.public: [],
            DataChannelType.private: []
        }
        
        # FIXME
        self.account = None
        self._ib_params_for_channels_subscription = {}
        
        self._sub_num = self._num_subscribed = 0
        
        self._background_task_freq = 10  # in seconds
        self._background_thread = None

        # since reqMktData will subscribe to bid/ask + last price/quantity automatically,
        # use this to save down which tick types the system has subscribed
        self._subscribed_market_data_tick_types = defaultdict(list)
        self._ib_thread = None

    def set_callback(self, callback: Callable[[str], Awaitable[None] | None], raw_msg: bool=False):
        '''
        Args:
            raw_msg: 
                if True, the callback will receive the raw messages.
                if False, the callback will receive parsed messages.
        '''
        self._callback = callback
        self._callback_raw_msg = raw_msg
    
    def add_account(self, account: IBAccount) -> IBAccount:
        if account.name not in self._accounts:
            self._accounts[account.name] = account
            self._logger.debug(f'added account {account}')
        else:
            raise ValueError(f'account name {account.name} has already been added')
        return account
        
    def add_product(self, product: IBProduct) -> IBProduct:
        if product.name not in self._products:
            self._products[product.name] = product
            self._logger.debug(f'added product {product.name}')
        else:
            existing_product = self._products[product.name]
            if existing_product != product:
                raise ValueError(f'product {product.name} has already been used for {existing_product}')
        return product
    
    def add_channel(self, channel: FullDataChannel, *, channel_type: Literal['public', 'private']): 
        channel_type: DataChannelType = DataChannelType[channel_type.lower()]
        if channel not in self._channels[channel_type]:
            self._channels[channel_type].append(channel)
            self._logger.debug(f'added {channel_type} channel {channel}')
    
    def _create_public_channel(self, product: IBProduct, resolution: Resolution):
        """Creates publich channel for internal use.
        Since IB's subscription does not require channel name,
        this function creates channel only for internal use, clarity and consistency.
        """
        self.add_product(product)
        if resolution.is_quote():
            channel = PublicDataChannel.orderbook
            echannel = self._adapter(channel.value, group='channel')
            orderbook_level = resolution.orderbook_level
            # TODO: how to handle orderbook_depth when received the orderbook data? shouldn't send the whole orderbook to engine's data object
            supported_orderbook_levels = self.SUPPORTED_ORDERBOOK_LEVELS
            if orderbook_level not in supported_orderbook_levels:
                raise NotImplementedError(f"{self.exch} ({channel}.{product.symbol}) orderbook_level={orderbook_level} is not supported, supported levels: {supported_orderbook_levels}")
            full_channel = '.'.join([echannel, product.symbol])
            # TODO
            # if 'num_rows' in kwargs:  # `num_rows` is a params in IB's reqMktDepth(...)
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
            full_channel = '.'.join([echannel, product.symbol, eresolution])
        else:
            raise NotImplementedError(f'{resolution=} is not supported for creating public channel')
        return full_channel

    def _create_private_channel(self, channel: PrivateDataChannel):
        channel = PrivateDataChannel[channel.lower()]
        return self._adapter(channel, group='channel')

    def _subscribe(self, channels: list[str], channel_type: DataChannelType):
        # TODO
        # ib_params = self._ib_params_for_channels_subscription[full_channel]
        for channel in channels:
            self._sub_num += 1
            if channel_type == DataChannelType.public:
                pass
                # if channel == 'kline':
                #     self._request_real_time_bar(**ib_params)
                
                # if channel == 'orderbook':
                #     if self._orderbook_level[pdt] == 1:
                #         if product.ptype not in self.ASSET_TYPES_WITHOUT_TICK_BY_TICK_DATA:
                #             tick_type = ib_params.get('tickType', 'BidAsk')
                #             assert tick_type in ['MidPoint', 'BidAsk'], f'tickType={tick_type} is not supported for trade channel'
                #             self._request_tick_by_tick_data(tick_type, **ib_params)
                #         else:
                #             self._request_market_data(**ib_params)
                #             self._subscribed_market_data_tick_types[pdt].extend([TickTypeEnum.BID, TickTypeEnum.BID_SIZE, TickTypeEnum.ASK, TickTypeEnum.ASK_SIZE])
                #     elif self._orderbook_level[pdt] == 2:
                #         self._request_market_depth(**ib_params)
                # elif channel == 'tradebook':
                #     if product.ptype not in self.ASSET_TYPES_WITHOUT_TICK_BY_TICK_DATA + self.ASSET_TYPES_WITHOUT_TICK_BY_TICK_LAST_DATA:
                #         tick_type = ib_params.get('tickType', 'Last')
                #         assert tick_type in ['Last', 'AllLast'], f'tickType={tick_type} is not supported for trade channel'
                #         self._request_tick_by_tick_data(tick_type, **ib_params)
                #     else:
                #         self._request_market_data(**ib_params)
                #         self._subscribed_market_data_tick_types[pdt].extend([TickTypeEnum.LAST, TickTypeEnum.LAST_SIZE])
                
                # if did not request market data but defined related params for it, request for it anyways
                # if not self._subscribed_market_data_tick_types[pdt] and \
                #     any(params in ib_params for params in ['genericTickList', 'snapshot', 'regulatorySnapshot']):
                #     self._request_market_data(**ib_params)
            else:
                if channel == 'account_update':
                    self._request_account_updates(acc)
                elif channel == 'account_summary':
                    self._request_account_summary(**ib_params)
    
    def _unsubscribe(self):
        self._sub_num = 0
        self._num_subscribed = 0
    
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
            zmq_msg = (1, 1, (self._bkr, product.exch, str(product), bids, asks, None, kwargs))
            self._zmq.send(*zmq_msg, receiver='engine')
        except:
            self._logger.exception(f'_update_orderbook exception ({position=} {operation=} {side=} {px=} {qty=} {kwargs=}):')
