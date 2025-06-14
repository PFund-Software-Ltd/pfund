from pfund.typing import tEnvironment
from pfund.accounts.account_base import BaseAccount


class IBAccount(BaseAccount):
    def __init__(
        self, 
        env: tEnvironment, 
        name: str='',
        host: str='', 
        port: int | None=None, 
        client_id: int | None=None, 
    ):
        super().__init__(env=env, bkr='IB', name=name)
        # FIXME: load from .env
        self._host = host
        self._port = port
        self._client_id = client_id
        assert host, f'`host` must be provided for {self.env} trading environment'
        assert port, f'`port` must be provided for {self.env} trading environment'
        assert client_id, f'`client_id` must be provided for {self.env} trading environment'
            