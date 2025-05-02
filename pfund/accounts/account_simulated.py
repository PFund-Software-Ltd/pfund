from pfund.enums import Environment, Broker
from pfund.accounts.account_base import BaseAccount


class SimulatedAccount(BaseAccount):
    def __init__(self, env: Environment, bkr: Broker, name: str=''):
        super().__init__(env, bkr, name=name)
