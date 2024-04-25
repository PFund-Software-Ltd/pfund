from pfund.brokers.broker_base import BaseBroker


def BacktestBroker(Broker) -> BaseBroker:
    class _BacktestBroker(Broker):
        def __init__(self):
            super().__init__(env='BACKTEST')
            self._initial_balances = None
            # TODO?
            # self.initial_positions = None
        
        # NOTE: in backtesting, although it allows to use the same broker and exchange objects,
        # it should take away their apis to prevent from calling them accidentally
        def _assert_no_apis(self):
            if hasattr(self, '_api'):
                self._api = None
            if hasattr(self, "exchanges"):
                for exchange in self.exchanges.values():
                    exchange._rest_api = None
                    exchange._ws_api = None

        def start(self):
            self._assert_no_apis()
            self.logger.debug(f'broker {self.name} started')

        def stop(self):
            self.logger.debug(f'broker {self.name} stopped')

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
        
        # TODO
        def initialize_positions(self):
            pass
        
        def get_initial_balances(self):
            return self._initial_balances
        
        # REVIEW
        def initialize_balances(self, account, initial_balances):
            if initial_balances is None:
                initial_balances = {'BTC': 10, 'USD': 1_000_000}  
            else: 
                initial_balances = {k.upper(): v for k, v in initial_balances.items()}
            self._initial_balances = initial_balances
            updates = {'ts': None, 'data': {k: {'wallet': v, 'available': v, 'margin': v} for k, v in initial_balances.items()}}
            trading_venue = account.exch if account.bkr == 'CRYPTO' else account.bkr
            self.pm.update_balances(trading_venue, account.name, updates)
        
        # TODO
        def place_orders(self, account, product, orders):
            pass
            # self.om.handle_msgs(...)
            # self.pm.handle_msgs(...)

        # TODO
        def cancel_orders(self, account, product, orders):
            pass
            # self.om.handle_msgs(...)
        
        # TODO
        def cancel_all_order(self):
            pass
            # self.om.handle_msgs(...)

        # TODO
        def amend_orders(self, account, product, orders):
            pass

    return _BacktestBroker()
