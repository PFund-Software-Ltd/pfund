from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, TypeAlias
if TYPE_CHECKING:
    from pfund.enums import PrivateDataChannel, PublicDataChannel
    from pfund._typing import tEnvironment, ProductName, AccountName
    from pfund.datas.resolution import Resolution
    from pfund.engines.trade_engine_settings import TradeEngineSettings
    from pfund.orders.order_base import BaseOrder
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
        from pfund.engines.trade_engine import TradeEngine

        self._env = Environment[env.upper()]
        self._logger: logging.Logger | None = None
        self._settings: TradeEngineSettings | None = getattr(TradeEngine, "_settings", None)
        
        self._products: defaultdict[ExchangeName, dict[ProductName, BaseProduct]] = defaultdict(dict)
        self._accounts: defaultdict[TradingVenue, dict[AccountName, BaseAccount]] = defaultdict(dict)

        self._order_manager = OrderManager(self)
        self._portfolio_manager = PortfolioManager(self)
    
    @classmethod
    def create_product(cls, basis: str, exch: str='', name: str='', symbol: str='', **specs) -> BaseProduct:
        from pfund.products import ProductFactory
        if cls.name == Broker.CRYPTO:
            assert exch in CryptoExchange.__members__, f'{exch} is not a valid crypto exchange'
            trading_venue = CryptoExchange[exch.upper()]
        else:
            trading_venue = cls.name
        Product = ProductFactory(trading_venue=trading_venue, basis=basis)
        return Product(basis=basis, exchange=exch, name=name, symbol=symbol, **specs)

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
    
    def start(self):
        # TODO: check if all product names and account names are unique
        self._add_default_private_channels()
        if self._settings.cancel_all_at['start']:
            self.cancel_all_orders(reason='start')
        self._logger.debug(f'broker {self._name} started')

    def stop(self):
        if self._settings.cancel_all_at['stop']:
            self.cancel_all_orders(reason='stop')
        self._logger.debug(f'broker {self._name} stopped')

    @abstractmethod
    def add_private_channel(self, channel: PrivateDataChannel):
        pass

    @abstractmethod
    def add_public_channel(self, channel: PublicDataChannel, product: BaseProduct, resolution: Resolution):
        pass
    
    # TODO: add more abstract methods, e.g. place_orders etc.
       
    @abstractmethod
    def create_order(self, *args, **kwargs) -> BaseOrder:
        pass
    
    @abstractmethod
    def place_orders(self, *args, **kwargs) -> list[BaseOrder]:
        pass

    @abstractmethod
    def cancel_all_orders(self, reason=None):
        pass
    
    def _add_default_private_channels(self):
        for channel in PrivateDataChannel:
            self.add_channel(channel, channel_type='private')

    # FIXME
    def distribute_msgs(self, channel, topic, info):
        if channel == 1:
            pass
        elif channel == 2:  # from api processes to data manager
            self._order_manager.handle_msgs(topic, info)
        elif channel == 3:
            self._portfolio_manager.handle_msgs(topic, info)
        # FIXME
        # elif channel == 4:  # from api processes to connection manager 
        #     self._connection_manager.handle_msgs(topic, info)
        #     if topic == 3 and self._settings.get('cancel_all_at', {}).get('disconnect', True):  # on disconnected
        #         self.cancel_all_orders(reason='disconnect')

    # FIXME: move to mtflow
    def schedule_jobs(self, scheduler: BackgroundScheduler):
        scheduler.add_job(self.reconcile_balances, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_positions, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_orders, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_trades, 'interval', seconds=10)
        for manager in [self._order_manager, self._portfolio_manager]:
            manager.schedule_jobs(scheduler)
