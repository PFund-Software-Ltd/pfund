from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.brokers.broker_base import BaseBroker

from pfund.enums import Environment, Broker
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings


def create_broker(env: Environment, bkr: Broker, settings: TradeEngineSettings | None=None) -> BaseBroker:
    if env in [Environment.BACKTEST, Environment.SANDBOX]:
        from pfund.brokers.broker_simulated import SimulatedBrokerFactory
        SimulatedBrokerClass: type[BaseBroker] = SimulatedBrokerFactory(bkr)
        broker = SimulatedBrokerClass(env)
    else:
        BrokerClass: type[BaseBroker] = Broker[bkr.upper()].broker_class
        broker = BrokerClass(env)
    if settings:
        broker.set_engine_settings(settings)
    return broker
