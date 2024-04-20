from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.core import tStrategy
    
from pfund.const.commons import SUPPORTED_CRYPTO_EXCHANGES
from pfund.mixins.backtest import BacktestMixin


# HACK: since python doesn't support dynamic typing, true return type should be subclass of BacktestMixin and tStrategy
# write -> BacktestMixin | tStrategy for better intellisense in IDEs 
def BacktestStrategy(Strategy: type[tStrategy], *args, **kwargs) -> BacktestMixin | tStrategy:
    class _BacktestStrategy(BacktestMixin, Strategy):
        def __init__(self, *_args, **_kwargs):
            try:
                Strategy.__init__(self, *_args, **_kwargs)
            except TypeError as e:
                raise TypeError(
                    f'if super().__init__() is called in {Strategy.__name__ }.__init__() (which is unnecssary), '
                    'make sure it is called with args and kwargs, i.e. super().__init__(*args, **kwargs)'
                ) from e
            self.initialize_mixin()
            self.add_strategy_signature(*_args, **_kwargs)

        # __getattr__ at this level to get the correct strategy name
        def __getattr__(self, attr):
            '''gets triggered only when the attribute is not found'''
            try:
                return super().__getattr__(attr)
            except AttributeError:
                class_name = Strategy.__name__
                raise AttributeError(f"'{class_name}' object or '{class_name}.data_tool' has no attribute '{attr}'")
        
        def to_dict(self):
            strategy_dict = super().to_dict()
            strategy_dict['class'] = Strategy.__name__
            strategy_dict['strategy_signature'] = self._strategy_signature
            strategy_dict['data_signatures'] = self._data_signatures
            return strategy_dict
        
        def _prepare_df(self):
            if self.name != '_dummy':
                return self.data_tool._prepare_df()
            
        def _prepare_df_with_models(self, *args, **kwargs):
            if self.name != '_dummy' and self.engine.mode == 'vectorized':
                self.data_tool._prepare_df_with_models(*args, **kwargs)
        
        def _append_to_df(self, *args, **kwargs):
            if self.engine.append_to_strategy_df and self.name != '_dummy':
                return self.data_tool._append_to_df(*args, **kwargs)
            
        def add_account(self, trading_venue: str, acc: str='', initial_balances: dict[str, int|float]|None=None, **kwargs):
            # NOTE: do NOT pass in kwargs to super().add_account(),
            # this can prevent any accidental credential leak during backtesting
            account = super().add_account(trading_venue, acc=acc)
            bkr = 'CRYPTO' if trading_venue in SUPPORTED_CRYPTO_EXCHANGES else trading_venue
            broker = self.get_broker(bkr)
            broker.initialize_balances(account, initial_balances)
        
    return _BacktestStrategy(*args, **kwargs)