from pfund.typing import tEnvironment
from pfund.brokers.broker_base import BaseBroker


# TODO
class DeFiBroker(BaseBroker):
    def __init__(self, env: tEnvironment='SANDBOX'):
        super().__init__(env, 'DEFI')
        