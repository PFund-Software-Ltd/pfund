from pfund._typing import tEnvironment
from pfund.brokers.broker_base import BaseBroker


# TODO
class DappBroker(BaseBroker):
    def __init__(self, env: tEnvironment='SANDBOX'):
        super().__init__(env, 'DAPP')
        