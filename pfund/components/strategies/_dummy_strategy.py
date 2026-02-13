from pfund.components.strategies.strategy_base import BaseStrategy


class DummyStrategy(BaseStrategy):
    name: str = '_dummy'
    
    # add event driven functions to dummy strategy to avoid NotImplementedError in backtesting
    def on_quote(self, *args, **kwargs):
        pass
    
    def on_tick(self, *args, **kwargs):
        pass

    def on_bar(self, *args, **kwargs):
        pass
