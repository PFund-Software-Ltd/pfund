from pfund.typing import tENVIRONMENT
from pfund.brokers.broker_trade import TradeBroker


# TODO
class DeFiBroker(TradeBroker):
    def __init__(self, env: tENVIRONMENT='SANDBOX'):
        super().__init__(env, 'DEFI')
        