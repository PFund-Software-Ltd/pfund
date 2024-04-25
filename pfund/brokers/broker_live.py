from pfund.brokers.broker_base import BaseBroker
from pfund.products.product_base import BaseProduct
from pfund.managers import ConnectionManager, DataManager, OrderManager, PortfolioManager, RiskManager


class LiveBroker(BaseBroker):
    def __init__(self, env, name):
        super().__init__(env, name)
        self._zmq = None
        self.connection_manager = self.cm = ConnectionManager(self)
        self.data_manager = self.dm = DataManager(self)
        self.order_manager = self.om = OrderManager(self)
        self.portfolio_manager = self.pm = PortfolioManager(self)
        self.risk_manager = self.rm = RiskManager(self)

    @property
    def balances(self):
        return self.portfolio_manager.balances[self.bkr] if self.bkr != 'CRYPTO' else self.portfolio_manager.balances
    
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

    def get_data(self, product: BaseProduct, resolution: str | None=None):
        return self.data_manager.get_data(product, resolution=resolution)
    
    def cancel_all_orders(self, reason=None):
        print(f'broker cancel_all_orders, reason={reason}')

    def add_listener(self, listener, listener_key, event_type):
        if event_type == 'public':
            # add listener for public events, e.g. quote, tick etc.
            self.data_manager.add_listener(listener, listener_key)
        else:
            # add listener for private events, e.g. order, trade, balance, position
            for manager in [self.rm, self.cm, self.om, self.pm]:
                manager.add_listener(listener, listener_key)
    
    def remove_listener(self, listener, listener_key, event_type):
        if event_type == 'public':
            # remove listener for public events, e.g. quote, tick etc.
            self.data_manager.remove_listener(listener, listener_key)
        else:
            # remove listener for private events, e.g. order, trade, balance, position
            for manager in [self.rm, self.cm, self.om, self.pm]:
                manager.remove_listener(listener, listener_key)

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