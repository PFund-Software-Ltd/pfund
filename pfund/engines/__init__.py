from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.engines.trade_engine import TradeEngine
    from pfund.engines.backtest_engine import BacktestEngine

import os

from pfund.enums import Environment


def get_engine() -> TradeEngine | BacktestEngine | None:
    from pfund.engines.trade_engine import TradeEngine
    from pfund.engines.backtest_engine import BacktestEngine
    '''Get the engine instance in use'''
    env = os.getenv('trading_env', None)
    if env is None or env not in Environment.__members__:
        return None
    env = Environment[env]
    if env == Environment.BACKTEST:
        return BacktestEngine()
    else:
        return TradeEngine()
