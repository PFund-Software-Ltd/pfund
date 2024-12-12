from pfund.accounts.account_base import BaseAccount
from pfund.const.enums import Environment, Broker


class IBAccount(BaseAccount):
    def __init__(self, env: Environment, host: str='', port: int=None, client_id: int=None, name: str='', **kwargs):
        super().__init__(env, Broker.IB, name=name, host=host, port=port, client_id=client_id, **kwargs)
        if self.env in [Environment.PAPER, Environment.LIVE]:
            assert host, f'`host` must be provided for {self.env} trading environment'
            assert port, f'`port` must be provided for {self.env} trading environment'
            assert client_id, f'`client_id` must be provided for {self.env} trading environment'
            