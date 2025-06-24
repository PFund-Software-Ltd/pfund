from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import tBroker
    from pfund.brokers.broker_base import BaseBroker

from pfund.enums import Environment, Broker


def create_broker(env: Environment | str, bkr: Broker | tBroker) -> BaseBroker:
    env = Environment[env.upper()]
    if env in [Environment.BACKTEST, Environment.SANDBOX]:
        from pfund.brokers.broker_simulated import SimulatedBrokerFactory
        SimulatedBroker = SimulatedBrokerFactory(bkr)
        broker = SimulatedBroker(env)
    else:
        BrokerClass = Broker[bkr.upper()].broker_class
        broker = BrokerClass(env)
    return broker