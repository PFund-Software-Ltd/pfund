from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.datas.data_time_based import TimeBasedData

from collections import defaultdict

from pfund.brokers.broker_trade import BaseBroker


def BacktestBrokerFactory(Broker: type[BaseBroker]) -> type[BaseBroker]:
    class BacktestBroker(Broker):
        _DEFAULT_INITIAL_BALANCES = {'BTC': 10, 'USD': 1_000_000}
        
        def __init__(self):
            super().__init__(env='BACKTEST')
            self._initial_balances = defaultdict(dict)  # {trading_venue: {acc1: balances_dict, acc2: balances_dict} }
            # TODO?
            # self.initial_positions = None
        
        def start(self):
            self.logger.debug(f'broker {self.name} started')
            self.initialize_balances()
            
        def stop(self):
            self.logger.debug(f'broker {self.name} stopped')
        
        def add_account(self, acc: str='', initial_balances: dict[str, int|float]|None=None, **kwargs):
            # NOTE: do NOT pass in kwargs to super().add_account(),
            # this can prevent any accidental credential leak during backtesting
            exch = kwargs.get('exch', '')
            strat = kwargs.get('strat', '')
            account_type = kwargs.get('account_type', '')
            account = super().add_account(acc=acc, exch=exch, strat=strat, account_type=account_type)
            if initial_balances is None:
                initial_balances = self._DEFAULT_INITIAL_BALANCES
            else: 
                initial_balances = {k.upper(): v for k, v in initial_balances.items()}
            trading_venue = account.exch if self.bkr == 'CRYPTO' else self.bkr
            self._initial_balances[trading_venue][account.name] = initial_balances
            return account

        def add_data_channel(self, *args, **kwargs):
            pass
        
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
        
        def get_initial_balances(self):
            return self._initial_balances
        
        def initialize_balances(self):
            for trading_venue in self._initial_balances:
                for acc, initial_balances in self._initial_balances[trading_venue].items():
                    updates = {'ts': None, 'data': {k: {'wallet': v, 'available': v, 'margin': v} for k, v in initial_balances.items()}}
                    self.portfolio_manager.update_balances(trading_venue, acc, updates)
        
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
        def cancel_all_order(self):
            pass
            # self.order_manager.handle_msgs(...)

        # TODO
        def amend_orders(self, account, product, orders):
            pass

    return BacktestBroker
