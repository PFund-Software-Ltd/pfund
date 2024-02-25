# Setup

The following setup is used across examples unless further specified:
```python
import pfund as pf

# quick setup to make other examples runnable
engine = pf.BacktestEngine()
DemoStrategy = type('DemoStrategy', (pf.Strategy,), {})
strategy = engine.add_strategy(DemoStrategy(), name='demo_strategy')
strategy.add_data('BYBIT', 'BTC', 'USDT', 'PERP', resolution='1d')
```
> The above code snippet will be referred in other sections as [setup]