from pfund.typing import tENVIRONMENT
from pfund.brokers.broker_trade import TradeBroker


# TODO
class DappBroker(TradeBroker):
    def __init__(self, env: tENVIRONMENT='SANDBOX'):
        super().__init__(env, 'DAPP')
        