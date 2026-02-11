# pyright: reportUninitializedInstanceVariable=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pfund.brokers.managers.order_manager import OrderManager
    from pfund.brokers.managers.portfolio_manager import PortfolioManager
    from pfund.typing import Currency, AccountName, ProductName
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.entities.products.product_base import BaseProduct
    from pfund.brokers.broker_base import BaseBroker
    from pfund.entities.accounts.account_base import BaseAccount

import logging
from decimal import Decimal

from pfund.enums import Environment, TradingVenue, Broker


def SimulatedBrokerFactory(broker: str) -> type[BaseBroker]:
    from pfund.enums import Broker
    BrokerClass = Broker[broker.upper()].broker_class
    return type("Simulated" + BrokerClass.__name__, (SimulatedBroker, BrokerClass), {"__module__": __name__})


# TODO: how to add margin calls?
class SimulatedBroker:
    # NOTE: host, port, client_id are required for using PAPER/LIVE trading data feeds in SANDBOX trading
    WHITELISTED_ACCOUNT_FIELDS: ClassVar[list[str]] = [
        '_env', 
        'trading_venue', 
        'name', 
        '_host',
        '_port',
        '_client_id',
    ]
    DEFAULT_INITIAL_BALANCES: ClassVar[dict[TradingVenue, dict[Currency, Decimal]]] = {
        TradingVenue.IBKR: {
            'USD': Decimal(1_000_000),
        },
        TradingVenue.BYBIT: {
            'BTC': Decimal(10),
            'USDT': Decimal(1_000_000),
        },
    }

    _logger: logging.Logger
    name: Broker
    _env: Environment
    _settings: BacktestEngineSettings
    _products: dict[TradingVenue, dict[ProductName, BaseProduct]]
    _accounts: dict[TradingVenue, dict[AccountName, BaseAccount]]
    _order_manager: OrderManager
    _portfolio_manager: PortfolioManager

    # TODO
    def _safety_check(self):
        # TODO: add a function to override all the existing functions in live broker
        pass
        
    def _accounts_check(self):
        accounts: list[BaseAccount] = [
            account
            for accounts_per_venue in self._accounts.values()
            for account in accounts_per_venue.values()
        ]

        for account in accounts:
            # remove all the attributes that are not in WHITELISTED_ACCOUNT_FIELDS
            for k, v in list(account.__dict__.items()):
                if v and k not in self.WHITELISTED_ACCOUNT_FIELDS:
                    self._logger.warning(f'removed non-whitelisted attribute {k} from {self.name} account {account.name}')
                    delattr(account, k)

        # FIXME: no longer needed? or only needs it for SANDBOX env?
        if self.name == Broker.IBKR:
            from pfund.entities.accounts.account_ibkr import IBKRAccount
            account = list(self._accounts.values())[0]  # IB has only one account for SANDBOX trading
            # create an IB account to check if host, port, client_id are provided, if not, it raises an error
            IBKRAccount(
                env=self._env, 
                name=account.name,
                host=getattr(account, 'host', ''),
                port=getattr(account, 'port', None),
                client_id=getattr(account, 'client_id', None),
            )

    def _initialize_balances(self):
        from pfund.entities.balances.balance_update import BalanceUpdate
        initial_balances = self._settings.initial_balances or self.DEFAULT_INITIAL_BALANCES
        for tv in initial_balances:
            balances = initial_balances[tv]
            accounts = self._accounts[tv]
            for ccy, amount in balances.items():  
                update = BalanceUpdate(
                    ts=None,
                    data={
                        # REVIEW: same amount for wallet, available, margin?
                        ccy: {'wallet': amount, 'available': amount, 'margin': amount}
                    }
                )
                for acc in accounts:
                    self.add_balance(tv, acc, ccy)
                    self._portfolio_manager.update_balances(tv, acc, update)

    def _initialize_positions(self):
        raise NotImplementedError('initial positions are not supported yet')
    
    def start(self):
        self._safety_check()
        self._accounts_check()
        self._logger.debug(f'broker {self.name} started')
        self._initialize_balances()
        # TODO: handle initial positions
        # if self._settings.initial_positions:
        #     self._initialize_positions()
        
    def stop(self):
        self._logger.debug(f'broker {self.name} stopped')
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_balances(self, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_positions(self, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_orders(self, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_trades(self, *args, **kwargs):
        pass
    
    # TODO
    def place_orders(self, account, product, orders):
        pass
        # self.order_manager.handle_msgs(...)
        # self.portfolio_manager.handle_msgs(...)

    # TODO
    def cancel_orders(self, account, product, orders):
        pass
        # self.order_manager.handle_msgs(...)
    
    # TODO
    def cancel_all_orders(self):
        pass
        # self.order_manager.handle_msgs(...)

    # TODO
    def amend_orders(self, account, product, orders):
        pass
