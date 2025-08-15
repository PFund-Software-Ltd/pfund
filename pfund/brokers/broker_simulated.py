from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.engines.backtest_engine_settings import BacktestEngineSettings
    from pfund.products.product_base import BaseProduct
    from pfund.brokers.broker_base import BaseBroker

from collections import defaultdict

from pfund.enums import Environment, Broker
from pfund.accounts.account_simulated import SimulatedAccount


def SimulatedBrokerFactory(broker: str) -> type[BaseBroker]:
    from pfund.enums import Broker
    BrokerClass = Broker[broker.upper()].broker_class
    return type("Simulated" + BrokerClass.__name__, (SimulatedBroker, BrokerClass), {"__module__": __name__})


# TODO: how to add margin calls?
class SimulatedBroker:
    # NOTE: host, port, client_id are required for using PAPER trading data feeds in SANDBOX trading
    WHITELISTED_ACCOUNT_FIELDS = ['_env', 'trading_venue', 'name', '_host', '_port', '_client_id']
    DEFAULT_INITIAL_BALANCES = {'BTC': 10, 'ETH': 100, 'USD': 1_000_000}
    
    def __init__(self: SimulatedBroker | BaseBroker, env: Literal['BACKTEST', 'SANDBOX']='BACKTEST'):
        super().__init__(env=env)
        if self._env == Environment.BACKTEST:
            from pfund.engines.backtest_engine import BacktestEngine
            self._settings: BacktestEngineSettings | None = getattr(BacktestEngine, "_settings", None)
        self._initial_balances = defaultdict(dict)  # {trading_venue: {acc1: balances_dict, acc2: balances_dict} }
        # TODO
        # self._initial_positions = None
    
    # TODO
    def _safety_check(self: SimulatedBroker | BaseBroker):
        # TODO: add a function to override all the existing functions in live broker
        pass
        
    def _accounts_check(self: SimulatedBroker | BaseBroker):
        assert all(isinstance(account, SimulatedAccount) for account in self._accounts.values()), 'all accounts must be SimulatedAccount'
        if self.name == Broker.IB:
            from pfund.accounts.account_ib import IBAccount
            account = list(self._accounts.values())[0]  # IB has only one account for SANDBOX trading
            # create an IB account to check if host, port, client_id are provided, if not, it raises an error
            IBAccount(
                env=self._env, 
                name=account.name,
                host=getattr(account, 'host', ''),
                port=getattr(account, 'port', None),
                client_id=getattr(account, 'client_id', None),
            )

    def start(self: SimulatedBroker | BaseBroker):
        self._safety_check()
        self._accounts_check()
        self._logger.debug(f'broker {self.name} started')
        self.initialize_balances()
        
    def stop(self: SimulatedBroker | BaseBroker):
        self._logger.debug(f'broker {self.name} stopped')
    
    def add_account(
        self: SimulatedBroker | BaseBroker, 
        name: str='', 
        initial_balances: dict[str, float] | None=None,
        initial_positions: dict[BaseProduct, float]|None=None,
        **kwargs,
    ):
        account = super().add_account(name=name, **kwargs)
        # remove all the attributes that are not in WHITELISTED_ACCOUNT_FIELDS
        for k, v in account.__dict__.items():
            if k not in self.WHITELISTED_ACCOUNT_FIELDS:
                delattr(account, k)
        # TODO: assign different initial balances for different broker accounts
        if initial_balances is None:
            initial_balances = self.DEFAULT_INITIAL_BALANCES
        else: 
            initial_balances = {k.upper(): v for k, v in initial_balances.items()}
        trading_venue = account.exch if self.name == Broker.CRYPTO else self.name
        self._initial_balances[trading_venue][account.name] = initial_balances
        return account

    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_balances(self: SimulatedBroker | BaseBroker, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_positions(self: SimulatedBroker | BaseBroker, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_orders(self: SimulatedBroker | BaseBroker, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_trades(self: SimulatedBroker | BaseBroker, *args, **kwargs):
        pass
    
    def get_initial_balances(self: SimulatedBroker | BaseBroker):
        return self._initial_balances
    
    def initialize_balances(self: SimulatedBroker | BaseBroker):
        for trading_venue in self._initial_balances:
            for acc, initial_balances in self._initial_balances[trading_venue].items():
                updates = {'ts': None, 'data': {k: {'wallet': v, 'available': v, 'margin': v} for k, v in initial_balances.items()}}
                self._portfolio_manager.update_balances(trading_venue, acc, updates)
    
    # TODO
    def place_orders(self: SimulatedBroker | BaseBroker, account, product, orders):
        pass
        # self.order_manager.handle_msgs(...)
        # self.portfolio_manager.handle_msgs(...)

    # TODO
    def cancel_orders(self: SimulatedBroker | BaseBroker, account, product, orders):
        pass
        # self.order_manager.handle_msgs(...)
    
    # TODO
    def cancel_all_orders(self: SimulatedBroker | BaseBroker):
        pass
        # self.order_manager.handle_msgs(...)

    # TODO
    def amend_orders(self: SimulatedBroker | BaseBroker, account, product, orders):
        pass
