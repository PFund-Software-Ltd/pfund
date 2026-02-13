from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar
if TYPE_CHECKING:
    from pfund.enums import PrivateDataChannel, PublicDataChannel
    from pfund.typing import ProductName, AccountName, Currency
    from pfund.datas.resolution import Resolution
    from pfund.entities.orders.order_base import BaseOrder
    from pfund.entities.positions.position_base import BasePosition
    from pfund.entities.balances.balance_base import BaseBalance
    from pfund.entities.products.product_base import BaseProduct
    from pfund.entities.accounts.account_base import BaseAccount

import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
from pfund.brokers.managers.order_manager import OrderManager
from pfund.brokers.managers.portfolio_manager import PortfolioManager
from pfund.enums import Environment, Broker, TradingVenue, CryptoExchange


class BaseBroker(ABC):
    name: ClassVar[Broker]

    def __init__(self, env: Environment | str, settings: TradeEngineSettings | None=None):
        self._env: Environment = Environment[env.upper()]
        self._logger: logging.Logger = logging.getLogger('pfund')
        self._settings: TradeEngineSettings = settings or TradeEngineSettings()

        self._products: defaultdict[CryptoExchange, dict[ProductName, BaseProduct]] = defaultdict(dict)
        self._accounts: defaultdict[TradingVenue, dict[AccountName, BaseAccount]] = defaultdict(dict)

        self._order_manager = OrderManager(self)
        self._portfolio_manager: PortfolioManager[BaseBalance, BasePosition] = PortfolioManager()

    @property
    def portfolio_manager(self):
        return self._portfolio_manager
    pm = portfolio_manager

    @property
    def order_manager(self):
        return self._order_manager
    om = order_manager

    @classmethod
    def create_product(cls, basis: str, exch: str='', name: str='', symbol: str='', **specs) -> BaseProduct:
        from pfund.entities.products import ProductFactory
        if cls.name == Broker.CRYPTO:
            assert exch in CryptoExchange.__members__, f'{exch} is not a valid crypto exchange'
            trading_venue = CryptoExchange[exch.upper()]
        else:
            trading_venue = cls.name
        Product = ProductFactory(trading_venue=trading_venue, basis=basis)
        return Product(basis=basis, exchange=exch, name=name, symbol=symbol, specs=specs)

    @abstractmethod
    def add_product(self, *args: Any, **kwargs: Any) -> BaseProduct:
        pass

    @abstractmethod
    def add_account(self, *args: Any, **kwargs: Any) -> BaseAccount:
        pass

    @property
    def products(self):
        return self._products

    @property
    def accounts(self):
        return self._accounts

    @property
    def balances(self) -> dict[TradingVenue, dict[AccountName, dict[Currency, BaseBalance]]]:
        return self._portfolio_manager._balances

    @property
    def positions(self) -> dict[TradingVenue, dict[AccountName, dict[ProductName, BasePosition]]]:
        return self._portfolio_manager._positions

    def opened_orders(self):
        return self._order_manager.opened_orders

    def submitted_orders(self):
        return self._order_manager.submitted_orders

    def closed_orders(self):
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
    def _distribute_msgs(self, channel, topic, info):
        if ...:
            self.add_order(...)
            self._order_manager.update_orders(...)
        elif ...:
            self.add_position(...)
            self._portfolio_manager.update_positions(...)
        elif ...:
            self.add_balance(...)
            self._portfolio_manager.update_balances(...)
        # elif channel == 4:  # from api processes to connection manager
        #     self._connection_manager.handle_msgs(topic, info)
        #     if topic == 3 and self._settings.get('cancel_all_at', {}).get('disconnect', True):  # on disconnected
        #         self.cancel_all_orders(reason='disconnect')

    # FIXME:
    def schedule_jobs(self, scheduler: BackgroundScheduler):
        scheduler.add_job(self.reconcile_balances, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_positions, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_orders, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_trades, 'interval', seconds=10)
        for manager in [self._order_manager, self._portfolio_manager]:
            manager.schedule_jobs(scheduler)
