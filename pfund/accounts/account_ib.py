from pfund._typing import tEnvironment
from pfund.accounts.account_base import BaseAccount
from pfund.enums import Environment


class IBAccount(BaseAccount):
    def __init__(
        self, 
        env: tEnvironment, 
        name: str='',
        host: str='', 
        port: int | None=None, 
        client_id: int | None=None, 
    ):
        super().__init__(env=env, trading_venue='IB', name=name)
        self._host = host
        self._port = port
        self._client_id = client_id
        if self._env in [Environment.PAPER, Environment.LIVE]:
            assert self._host, f'`host` must be provided for {self._env} trading environment'
            assert self._port, f'`port` must be provided for {self._env} trading environment'
            assert self._client_id, f'`client_id` must be provided for {self._env} trading environment'
