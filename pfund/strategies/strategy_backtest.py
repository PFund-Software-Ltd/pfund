from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import StrategyT
    from pfund.data_tools import data_tool_backtest
    
from pfund.mixins.backtest_mixin import BacktestMixin


# HACK: since python doesn't support dynamic typing, true return type should be subclass of BacktestMixin and StrategyT
# write -> BacktestMixin | StrategyT for better intellisense in IDEs 
def BacktestStrategy(Strategy: type[StrategyT], *args, **kwargs) -> BacktestMixin | StrategyT:
    class _BacktestStrategy(BacktestMixin, Strategy):
        def __getattr__(self, name):
            if hasattr(super(), name):
                return getattr(super(), name)
            else:
                class_name = Strategy.__name__
                raise AttributeError(f"'{class_name}' object has no attribute '{name}'")
        
        # REVIEW
        def backtest(self, df: data_tool_backtest.BacktestDataFrame):
            raise Exception(f'Strategy "{self.name}" does not have a backtest() method, cannot run vectorized backtesting')
        
        def add_account(self, trading_venue: str, acc: str='', initial_balances: dict[str, int|float]|None=None, **kwargs):
            return super().add_account(trading_venue, acc=acc, initial_balances=initial_balances, **kwargs)
            
        def add_strategy(self, strategy: StrategyT, name: str='') -> BacktestMixin | StrategyT:
            strategy = BacktestStrategy(type(strategy), *strategy._args, **strategy._kwargs)
            return super().add_strategy(strategy, name=name)
        
        def on_start(self):
            if not self.accounts:
                for trading_venue in self.get_trading_venues():
                    if trading_venue == 'BYBIT':
                        kwargs = {'account_type': 'UNIFIED'}
                    else:
                        kwargs = {}
                    account = self.add_account(trading_venue, **kwargs)
                    broker = self.get_broker(account.bkr)
                    broker.initialize_balances()
            # TODO
            if self._is_signal_df_required and self._signal_df is None:
                self.logger.warning(f"creating signal_df for '{self.name}' on the fly")
                # signal_df: pd.DataFrame | pl.LazyFrame = self.flow()
                # self._set_signal_df(signal_df)
            super().on_start()
        
        # TODO
        def load(self):
            pass
        
        # TODO
        def dump(self):
            pass
        
    try: 
        return _BacktestStrategy(*args, **kwargs)
    except TypeError as e:
        raise TypeError(
            f'if super().__init__() is called in {Strategy.__name__ }.__init__() (which is unnecssary), '
            'make sure it is called with args and kwargs, i.e. super().__init__(*args, **kwargs)'
        ) from e