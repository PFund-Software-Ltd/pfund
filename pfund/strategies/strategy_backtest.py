from pfund.strategies.strategy_base import BaseStrategy
from pfund.const.commons import *

        
def BacktestStrategy(Strategy: BaseStrategy, *args, **kwargs) -> BaseStrategy:
    from pfund.mixins.backtest import BacktestMixin
    
    class _BacktestStrategy(BacktestMixin, Strategy):
        # __getattr__ at this level to get the correct strategy name
        def __getattr__(self, attr):
            '''gets triggered only when the attribute is not found'''
            try:
                return super().__getattr__(attr)
            except AttributeError:
                class_name = Strategy.__name__
                raise AttributeError(f"'{class_name}' object or '{class_name}.data_tool' has no attribute '{attr}', make sure super().__init__() is called in your strategy {class_name}.__init__()")
        
        def _prepare_df(self):
            if self.name != '_dummy':
                return self.data_tool._prepare_df()
            
        def _prepare_df_with_models(self, *args, **kwargs):
            if self.name != '_dummy' and self._Engine.mode == 'vectorized':
                self.data_tool._prepare_df_with_models(*args, **kwargs)
        
        def _append_to_df(self, *args, **kwargs):
            if self._Engine.append_to_strategy_df and self.name != '_dummy':
                return self.data_tool._append_to_df(*args, **kwargs)
            
        def add_account(self, trading_venue: str, acc: str='', initial_balances: dict[str, int|float]|None=None, **kwargs):
            # NOTE: do NOT pass in kwargs to super().add_account(),
            # this can prevent any accidental credential leak during backtesting
            account = super().add_account(trading_venue, acc=acc)
            bkr = 'CRYPTO' if trading_venue in SUPPORTED_CRYPTO_EXCHANGES else trading_venue
            broker = self.get_broker(bkr)
            broker.initialize_balances(account, initial_balances)
        
    return _BacktestStrategy(*args, **kwargs)