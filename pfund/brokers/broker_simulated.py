from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.engines.backtest_engine_settings import BacktestEngineSettings
    from pfund.products.product_base import BaseProduct
    from pfund.brokers.broker_trade import TradeBroker

from collections import defaultdict

from pfund.enums import Environment, Broker
from pfund.accounts.account_simulated import SimulatedAccount


def SimulatedBrokerFactory(broker: str) -> type[TradeBroker]:
    from pfund.enums import Broker
    BrokerClass = Broker[broker.upper()].broker_class
    return type("Simulated" + BrokerClass.__name__, (SimulatedBroker, BrokerClass), {"__module__": __name__})


# TODO: how to add margin calls?
class SimulatedBroker:
    DEFAULT_INITIAL_BALANCES = {'BTC': 10, 'ETH': 100, 'USD': 1_000_000}
    
    def __init__(self: SimulatedBroker | TradeBroker, env: Literal['BACKTEST', 'SANDBOX']='BACKTEST'):
        super().__init__(env=env)
        if self._env == Environment.BACKTEST:
            from pfund.engines.backtest_engine import BacktestEngine
            self._run_mode = BacktestEngine._run_mode
            self._settings: BacktestEngineSettings = BacktestEngine._settings
        self._initial_balances = defaultdict(dict)  # {trading_venue: {acc1: balances_dict, acc2: balances_dict} }
        # TODO
        # self._initial_positions = None
    
    # TODO
    def _safety_check(self: SimulatedBroker | TradeBroker):
        assert all(isinstance(account, SimulatedAccount) for account in self._accounts.values()), 'all accounts must be SimulatedAccount'
        # TODO: add a function to override all the existing functions in live broker

    def start(self: SimulatedBroker | TradeBroker):
        self._safety_check()
        self._logger.debug(f'broker {self._name} started')
        self.initialize_balances()
        
    def stop(self: SimulatedBroker | TradeBroker):
        self._logger.debug(f'broker {self._name} stopped')
    
    def _create_account(self: SimulatedBroker | TradeBroker, name: str, **kwargs):
        # TODO: add initial_balances, initial_positions to SimulatedAccount, should not be broker's attributes
        return SimulatedAccount(env=self._env, bkr=self._name, name=name)
    
    def add_account(
        self: SimulatedBroker | TradeBroker, 
        name: str='', 
        initial_balances: dict[str, float] | None=None,
        initial_positions: dict[BaseProduct, float]|None=None,
    ):
        account = super().add_account(name=name)
        if initial_balances is None:
            initial_balances = self.DEFAULT_INITIAL_BALANCES
        else: 
            initial_balances = {k.upper(): v for k, v in initial_balances.items()}
        trading_venue = account.exch if self._name == Broker.CRYPTO else self._name
        self._initial_balances[trading_venue][account.name] = initial_balances
        return account

    def add_data_channel(self: SimulatedBroker | TradeBroker, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_balances(self: SimulatedBroker | TradeBroker, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_positions(self: SimulatedBroker | TradeBroker, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_orders(self: SimulatedBroker | TradeBroker, *args, **kwargs):
        pass
    
    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_trades(self: SimulatedBroker | TradeBroker, *args, **kwargs):
        pass
    
    def get_initial_balances(self: SimulatedBroker | TradeBroker):
        return self._initial_balances
    
    def initialize_balances(self: SimulatedBroker | TradeBroker):
        for trading_venue in self._initial_balances:
            for acc, initial_balances in self._initial_balances[trading_venue].items():
                updates = {'ts': None, 'data': {k: {'wallet': v, 'available': v, 'margin': v} for k, v in initial_balances.items()}}
                self._portfolio_manager.update_balances(trading_venue, acc, updates)
    
    # TODO
    def place_orders(self: SimulatedBroker | TradeBroker, account, product, orders):
        pass
        # self.order_manager.handle_msgs(...)
        # self.portfolio_manager.handle_msgs(...)

    # TODO
    def cancel_orders(self: SimulatedBroker | TradeBroker, account, product, orders):
        pass
        # self.order_manager.handle_msgs(...)
    
    # TODO
    def cancel_all_orders(self: SimulatedBroker | TradeBroker):
        pass
        # self.order_manager.handle_msgs(...)

    # TODO
    def amend_orders(self: SimulatedBroker | TradeBroker, account, product, orders):
        pass
