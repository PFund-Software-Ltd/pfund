from pfund.typing import tEnvironment
from pfund.brokers.broker_trade import TradeBroker


# TODO
class DappBroker(TradeBroker):
    def __init__(self, env: tEnvironment='SANDBOX'):
        super().__init__(env, 'DAPP')
        