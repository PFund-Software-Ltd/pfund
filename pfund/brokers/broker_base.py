from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler
    from pfeed.enums import DataSource
    from pfeed.typing import tDATA_SOURCE
    from pfeed.feeds.market_feed import MarketFeed
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.products.product_base import BaseProduct
    from pfund.brokers.broker_crypto import CryptoBroker
    from pfund.brokers.ib.broker_ib import IBBroker
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.typing import tENVIRONMENT, tBROKER
    from pfund.typing import TradeEngineSettingsDict

import logging
from collections import defaultdict

from pfund.datas.data_base import BaseData
from pfund.managers.connection_manager import ConnectionManager
from pfund.managers.data_manager import DataManager
from pfund.managers.order_manager import OrderManager
from pfund.managers.portfolio_manager import PortfolioManager
from pfund.managers.risk_manager import RiskManager
from pfund.enums import Environment, Broker, PublicDataChannel, PrivateDataChannel, DataChannelType
from pfund.engines import get_engine


class BaseBroker:
    def __init__(self, env: tENVIRONMENT, name: tBROKER):
        self.env = Environment[env.upper()]
        if self.env == Environment.BACKTEST:
            assert self.__class__.__name__ == '_BacktestBroker', f'env={self.env} is only allowed to be created using _BacktestBroker'
        self.name = self.bkr = Broker[name.upper()]
        self.logger = logging.getLogger('pfund')
        
        self._engine = get_engine()
        self._zmq = None
        self._settings: TradeEngineSettingsDict = {}
        
        self._products = defaultdict(dict)  # {exch: {pdt1: product1, pdt2: product2, exch1_pdt3: product, exch2_pdt3: product} }
        self._accounts = defaultdict(dict)  # {trading_venue: {acc1: account1, acc2: account2} }
    
        self._data_feed: MarketFeed | None = None
        self.connection_manager = ConnectionManager(self)
        self.data_manager = DataManager(self)
        self.order_manager = OrderManager(self)
        self.portfolio_manager = PortfolioManager(self)
        self.risk_manager = RiskManager(self)
    
    # TODO: use other data source, e.g. databento, only support TradFi Broker
    def use_data_source(self, data_source: tDATA_SOURCE | DataSource):
        from pfund.engines import TradeEngine
        self._data_feed: MarketFeed = TradeEngine.get_feed(data_source, use_deltalake=True)
        # TODO: create feed for streaming and somehow pass it to connection manager
        
    @property
    def products(self):
        return self._products
    
    @property
    def accounts(self):
        return self._accounts

    @property
    def balances(self):
        return self.portfolio_manager.balances[self.bkr.value] if self.bkr != Broker.CRYPTO else self.portfolio_manager.balances
    
    @property
    def positions(self):
        return self.portfolio_manager.positions
    
    @property
    def orders(self, type_='opened'):
        if type_ == 'opened':
            return self.order_manager.opened_orders
        elif type_ == 'submitted':
            return self.order_manager.submitted_orders
        elif type_ == 'closed':
            return self.order_manager.closed_orders
    
    def start(self, zmq=None):
        from pfund.engines import TradeEngine
        self._settings = TradeEngine.settings
        self._zmq = zmq
        self.connection_manager.connect()
        if self._settings.get('cancel_all_at', {}).get('start', True):
            self.cancel_all_orders(reason='start')
        self.logger.debug(f'broker {self.name} started')

    def stop(self):
        self._zmq = None
        if self._settings.get('cancel_all_at', {}).get('stop', True):
            self.cancel_all_orders(reason='stop')
        self.connection_manager.disconnect()
        self.logger.debug(f'broker {self.name} stopped')

    def get_zmq(self):
        return self._zmq

    def get_data(self, product: BaseProduct, resolution: str) -> BaseData | None:
        return self.data_manager.get_data(product, resolution=resolution)
    
    def cancel_all_orders(self, reason=None):
        print(f'broker cancel_all_orders, reason={reason}')

    def _create_public_data_channel(self, data: TimeBasedData) -> PublicDataChannel:
        if data.is_quote():
            channel = PublicDataChannel.orderbook
        elif data.is_tick():
            channel = PublicDataChannel.tradebook
        elif data.is_bar():
            channel = PublicDataChannel.candlestick
        else:
            raise ValueError(f'unknown data type: {data}')
        return channel
    
    def _create_data_channel_type(
        self, 
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: Literal['public', 'private']=''
    ) -> DataChannelType:
        if channel in [PublicDataChannel, PrivateDataChannel]:
            channel_type = DataChannelType.public if channel in PublicDataChannel else DataChannelType.private
        else:
            assert channel_type, 'channel_type "public" or "private" must be provided'
            channel_type = DataChannelType[channel_type.upper()]
        return channel_type
    
    def _add_data_listener(self, listener: BaseStrategy, data: BaseData):
        self.data_manager._add_listener(listener, data)

    def _remove_data_listener(self, listener: BaseStrategy, data: BaseData):
        self.data_manager._remove_listener(listener, data)
    
    def distribute_msgs(self, channel, topic, info):
        if channel == 1:
            self.data_manager.handle_msgs(topic, info)
        elif channel == 2:  # from api processes to data manager
            self.order_manager.handle_msgs(topic, info)
        elif channel == 3:
            self.portfolio_manager.handle_msgs(topic, info)
        elif channel == 4:  # from api processes to connection manager 
            self.connection_manager.handle_msgs(topic, info)
            if topic == 3 and self._settings.get('cancel_all_at', {}).get('disconnect', True):  # on disconnected
                self.cancel_all_orders(reason='disconnect')

    def schedule_jobs(self: CryptoBroker | IBBroker, scheduler: BackgroundScheduler):
        scheduler.add_job(self.reconcile_balances, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_positions, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_orders, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_trades, 'interval', seconds=10)
        for manager in [self.risk_manager, self.connection_manager, self.order_manager, self.portfolio_manager, self.data_manager]:
            manager.schedule_jobs(scheduler)