import os
import pytest

import pfund as pf
from pfund import BacktestEngine
from pfund.const.enums import Environment


@pytest.mark.smoke
def test_init(mocker):
    mocker.spy(BacktestEngine, '__new__')
    mocker.spy(BacktestEngine, '__init__')
    engine = BacktestEngine()
    assert BacktestEngine.__new__.call_count == 1
    assert BacktestEngine.__init__.call_count == 1
    assert BacktestEngine.env == Environment.BACKTEST
    assert os.getenv('env') == 'BACKTEST'
    assert engine._initialized is True

@pytest.mark.smoke
def test_singleton():
    engine1 = BacktestEngine()
    engine2 = BacktestEngine()
    assert engine1 is engine2

@pytest.mark.smoke
def test_add_strategy():
    engine = BacktestEngine()
    class FakeStrategy(pf.Strategy):
        pass
    args = (1, 2, 3)
    kwargs = {'a': 1, 'b': 2, 'c': 3}
    fake_strategy = FakeStrategy(*args, **kwargs)
    name = 'fake_strategy'
    backtest_strategy = engine.add_strategy(fake_strategy, name=name)
    assert backtest_strategy.name == name
    assert backtest_strategy._args == fake_strategy._args == args
    assert backtest_strategy._kwargs == fake_strategy._kwargs == kwargs