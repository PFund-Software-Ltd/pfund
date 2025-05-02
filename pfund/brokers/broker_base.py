from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.typing import tENVIRONMENT, tBROKER

import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from pfund.datas.data_base import BaseData
from pfund.managers.data_manager import DataManager
from pfund.managers.order_manager import OrderManager
from pfund.managers.portfolio_manager import PortfolioManager
from pfund.enums import Environment, Broker


class BaseBroker(ABC):
    def __init__(self, env: tENVIRONMENT, name: tBROKER):
        self._env = Environment[env.upper()]
        self._name = Broker[name.upper()]
        self._logger = logging.getLogger('pfund')
        
        self._zmq = None
        self._products = defaultdict(dict)  # {exch: {pdt1: product1, pdt2: product2, exch1_pdt3: product, exch2_pdt3: product} }
        self._accounts = defaultdict(dict)  # {trading_venue: {acc1: account1, acc2: account2} }
    
        self._order_manager = OrderManager(self)
        self._portfolio_manager = PortfolioManager(self)
        # FIXME: move to databoy
        self._data_manager = DataManager(self)
    
    @property
    def name(self):
        return self._name
    
    @property
    def products(self):
        return self._products
    
    @property
    def accounts(self):
        return self._accounts

    @property
    def balances(self):
        return self._portfolio_manager.balances[self._name] if self._name != Broker.CRYPTO else self._portfolio_manager.balances
    
    @property
    def positions(self):
        return self._portfolio_manager.positions
    
    @property
    def orders(self, type_='opened'):
        if type_ == 'opened':
            return self._order_manager.opened_orders
        elif type_ == 'submitted':
            return self._order_manager.submitted_orders
        elif type_ == 'closed':
            return self._order_manager.closed_orders
    
    @abstractmethod
    def start(self, zmq=None):
        pass

    @abstractmethod
    def stop(self):
        pass
    
    # TODO: add more abstract methods, e.g. place_orders etc.

    @abstractmethod
    def cancel_all_orders(self, reason=None):
        pass
    
    # FIXME: move to databoy
    def get_data(self, product: BaseProduct, resolution: str) -> BaseData | None:
        return self._data_manager.get_data(product, resolution=resolution)
    
    # FIXME: move to databoy
    def _add_data_listener(self, listener: BaseStrategy, data: BaseData):
        self._data_manager._add_listener(listener, data)

    # FIXME: move to databoy
    def _remove_data_listener(self, listener: BaseStrategy, data: BaseData):
        self._data_manager._remove_listener(listener, data)
    