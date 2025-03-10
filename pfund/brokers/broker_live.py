from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.datas.data_base import BaseData
    from pfund.products.product_base import BaseProduct
    from pfund.typing.literals import tENVIRONMENT

from pfund.brokers.broker_base import BaseBroker
from pfund.managers.connection_manager import ConnectionManager
from pfund.managers.data_manager import DataManager
from pfund.managers.order_manager import OrderManager
from pfund.managers.portfolio_manager import PortfolioManager
from pfund.managers.risk_manager import RiskManager
from pfund.enums import Broker, PublicDataChannel, PrivateDataChannel, DataChannelType


class LiveBroker(BaseBroker):
    def __init__(self, env: tENVIRONMENT, name: str):
        super().__init__(env, name)
        self._zmq = None
        self.connection_manager = self.cm = ConnectionManager(self)
        self.data_manager = self.dm = DataManager(self)
        self.order_manager = self.om = OrderManager(self)
        self.portfolio_manager = self.pm = PortfolioManager(self)
        self.risk_manager = self.rm = RiskManager(self)

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

    def _add_listener(self, listener, listener_key, event_type):
        if event_type == 'public':
            # add listener for public events, e.g. quote, tick etc.
            self.data_manager._add_listener(listener, listener_key)
        else:
            # add listener for private events, e.g. order, trade, balance, position
            for manager in [self.rm, self.cm, self.om, self.pm]:
                manager._add_listener(listener, listener_key)
    
    def _remove_listener(self, listener, listener_key, event_type):
        if event_type == 'public':
            # remove listener for public events, e.g. quote, tick etc.
            self.data_manager._remove_listener(listener, listener_key)
        else:
            # remove listener for private events, e.g. order, trade, balance, position
            for manager in [self.rm, self.cm, self.om, self.pm]:
                manager._remove_listener(listener, listener_key)
    
    def _create_public_data_channel(self, data: BaseData) -> PublicDataChannel | None:
        if not data.is_time_based():
            raise NotImplementedError('Only time-based data is supported for now')
        if data.is_resamplee():
            return None
        timeframe = data.timeframe
        if timeframe.is_quote():
            channel = PublicDataChannel.orderbook
        elif timeframe.is_tick():
            channel = PublicDataChannel.tradebook
        else:
            channel = PublicDataChannel.kline
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
    
    def distribute_msgs(self, channel, topic, info):
        if channel == 1:
            self.dm.handle_msgs(topic, info)
        elif channel == 2:  # from api processes to data manager
            self.om.handle_msgs(topic, info)
        elif channel == 3:
            self.pm.handle_msgs(topic, info)
        elif channel == 4:  # from api processes to connection manager 
            self.cm.handle_msgs(topic, info)
            if topic == 3 and self._settings.get('cancel_all_at', {}).get('disconnect', True):  # on disconnected
                self.cancel_all_orders(reason='disconnect')

    def run_regular_tasks(self):
        self.reconcile_balances()
        self.reconcile_positions()
        self.reconcile_orders()
        self.reconcile_trades()
        for manager in [self.rm, self.cm, self.om, self.pm, self.dm]:
            manager.run_regular_tasks()