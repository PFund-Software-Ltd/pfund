from pfund.strategies.strategy_base import BaseStrategy


class _DummyStrategy(BaseStrategy):
    # add event driven functions to dummy strategy to avoid NotImplementedError in backtesting
    def on_quote(self, *args, **kwargs):
        pass
    
    def on_tick(self, *args, **kwargs):
        pass

    def on_bar(self, *args, **kwargs):
        pass
