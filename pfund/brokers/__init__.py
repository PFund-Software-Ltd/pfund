from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.brokers.broker_base import BaseBroker
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.enums import Environment, Broker


def create_broker(env: Environment, bkr: Broker, settings: TradeEngineSettings | BacktestEngineSettings | None=None) -> BaseBroker:
    if env in [Environment.BACKTEST, Environment.SANDBOX]:
        from pfund.brokers.broker_simulated import SimulatedBrokerFactory
        if settings:
            assert isinstance(settings, BacktestEngineSettings), "settings must be a BacktestEngineSettings"
        SimulatedBrokerClass: type[BaseBroker] = SimulatedBrokerFactory(bkr)
        broker = SimulatedBrokerClass(env, settings=settings)  # pyright: ignore[reportArgumentType]
    else:
        if settings:
            assert isinstance(settings, TradeEngineSettings), "settings must be a TradeEngineSettings"
        BrokerClass: type[BaseBroker] = Broker[bkr.upper()].broker_class
        broker = BrokerClass(env, settings=settings)
    return broker