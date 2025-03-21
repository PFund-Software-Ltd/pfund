import os

from pfund.enums import Environment
from pfund.engines.trade_engine import TradeEngine
from pfund.engines.backtest_engine import BacktestEngine


def get_engine() -> TradeEngine | BacktestEngine | None:
    '''Get the engine instance in use'''
    env = os.getenv('trading_env', None)
    if env is None or env not in Environment.__members__:
        return None
    env = Environment[env]
    if env == Environment.BACKTEST:
        return BacktestEngine()
    else:
        return TradeEngine()
