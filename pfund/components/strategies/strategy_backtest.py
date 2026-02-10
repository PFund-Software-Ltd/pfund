from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
if TYPE_CHECKING:
    from pfund.typing import StrategyT
    from pfund._backtest.typing import BacktestDataFrame
    
from pfund._backtest.backtest_mixin import BacktestMixin


def BacktestStrategy(Strategy: type[StrategyT], *args: Any, **kwargs: Any) -> StrategyT:
    class _BacktestStrategy(BacktestMixin, Strategy):
        def __getattr__(self, name: str) -> Any:
            if hasattr(super(), name):
                return getattr(super(), name)
            else:
                class_name = Strategy.__name__
                raise AttributeError(f"'{class_name}' object has no attribute '{name}'")
        
        def backtest(self, df: BacktestDataFrame):  # pyright: ignore[reportUnusedParameter]
            raise Exception(f'Strategy "{self.name}" does not have a backtest() method, cannot run vectorized backtesting')
        
        def add_strategy(self, strategy: StrategyT, name: str='') -> StrategyT:
            strategy: StrategyT = BacktestStrategy(type(strategy), *strategy.__pfund_args__, **strategy.__pfund_kwargs__)
            return super().add_strategy(strategy, name=name)
        
        def add_accounts(self):
            super().add_accounts()
            if self.accounts:
                return
            # add account to each trading venue if no accounts are added
            trading_venues = set(product.trading_venue for product in self.products.values())
            for trading_venue in trading_venues:
                self.add_account(trading_venue=trading_venue)
        
        def on_start(self):
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
        return cast("StrategyT", _BacktestStrategy(*args, **kwargs))
    except TypeError as e:
        raise TypeError(
            f'if super().__init__() is called in {Strategy.__name__ }.__init__() (which is unnecssary), ' + 
            'make sure it is called with args and kwargs, i.e. super().__init__(*args, **kwargs)'
        ) from e