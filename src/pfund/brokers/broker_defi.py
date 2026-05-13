from pfund.brokers.broker_base import BaseBroker
from pfund.typing import tEnvironment


# TODO
class DeFiBroker(BaseBroker):
    def __init__(self, env: tEnvironment = "SANDBOX"):
        super().__init__(env, "DEFI")
