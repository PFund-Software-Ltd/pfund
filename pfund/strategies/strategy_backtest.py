from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.core import tStrategy
    
from pfund.mixins.backtest import BacktestMixin


# HACK: since python doesn't support dynamic typing, true return type should be subclass of BacktestMixin and tStrategy
# write -> BacktestMixin | tStrategy for better intellisense in IDEs 
def BacktestStrategy(Strategy: type[tStrategy], *args, **kwargs) -> BacktestMixin | tStrategy:
    class _BacktestStrategy(BacktestMixin, Strategy):
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
        
        def add_account(self, trading_venue: str, acc: str='', initial_balances: dict[str, int|float]|None=None, **kwargs):
            return super().add_account(trading_venue, acc=acc, initial_balances=initial_balances, **kwargs)
            
        def _is_dummy_strategy(self):
            return self.name == '_dummy'
        
        def start(self):
            super().start()
            if not self.accounts:
                for trading_venue in self.products:
                    if trading_venue == 'BYBIT':
                        kwargs = {'account_type': 'UNIFIED'}
                    else:
                        kwargs = {}
                    account = self.add_account(trading_venue, **kwargs)
                    broker = self.get_broker(account.bkr)
                    broker.initialize_balances()
        
        def clear_dfs(self):
            assert self.engine.mode == 'event_driven'
            if not self._is_dummy_strategy():
                self._data_tool.clear_df()
            for strategy in self.strategies.values():
                strategy.clear_dfs()
            for model in self.models.values():
                model.clear_dfs()
        
        def _add_raw_df(self, data, df):
            if not self._is_dummy_strategy():
                return self._data_tool.add_raw_df(data, df)
        
        def _set_data_periods(self, datas, **kwargs):
            if not self._is_dummy_strategy():
                return self._data_tool.set_data_periods(datas, **kwargs)
        
        def _prepare_df(self):
            if not self._is_dummy_strategy():
                return self._data_tool.prepare_df()
            
        def _append_to_df(self, **kwargs):
            if self.engine.append_signals and not self._is_dummy_strategy():
                return self._data_tool.append_to_df(self.data, self.predictions, **kwargs)
    
    try: 
        return _BacktestStrategy(*args, **kwargs)
    except TypeError as e:
        raise TypeError(
            f'if super().__init__() is called in {Strategy.__name__ }.__init__() (which is unnecssary), '
            'make sure it is called with args and kwargs, i.e. super().__init__(*args, **kwargs)'
        ) from e