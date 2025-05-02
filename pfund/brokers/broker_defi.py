from typing import Literal

from pfund.brokers.broker_trade import BaseBroker


# TODO
class DeFiBroker(BaseBroker):
    def __init__(self, env: Literal['SANDBOX', 'PAPER', 'LIVE']='SANDBOX'):
        super().__init__(env, 'DEFI')
        