from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, TypeAlias
if TYPE_CHECKING:
    from pfund.enums import PrivateDataChannel, PublicDataChannel
    from pfund.typing import tEnvironment, ProductName, AccountName
    from pfund.datas.resolution import Resolution
    from pfund.products.product_base import BaseProduct
    from pfund.accounts.account_base import BaseAccount

import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from pfund.managers.order_manager import OrderManager
from pfund.managers.portfolio_manager import PortfolioManager
from pfund.enums import Environment, Broker, TradingVenue, CryptoExchange


ExchangeName: TypeAlias = CryptoExchange | str


class BaseBroker(ABC):
    name: ClassVar[Broker]

    def __init__(self, env: Environment | tEnvironment):
        self._env = Environment[env.upper()]
        self._logger = logging.getLogger('pfund')
        
        self._products: defaultdict[ExchangeName, dict[ProductName, BaseProduct]] = defaultdict(dict)
        self._accounts: defaultdict[TradingVenue, dict[AccountName, BaseAccount]] = defaultdict(dict)

        self._order_manager = OrderManager(self)
        self._portfolio_manager = PortfolioManager(self)
    
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

    @abstractmethod
    def add_private_channel(self, *args, channel: PrivateDataChannel):
        pass

    @abstractmethod
    def add_public_channel(self, *args, channel: PublicDataChannel, product: BaseProduct, resolution: Resolution):
        pass
    
    # TODO: add more abstract methods, e.g. place_orders etc.

    @abstractmethod
    def cancel_all_orders(self, reason=None):
        pass
    