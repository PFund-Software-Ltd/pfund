from pfund.accounts.account_base import BaseAccount
from pfund.enums import Environment, Broker


class IBAccount(BaseAccount):
    def __init__(
        self, 
        env: Environment, 
        name: str='',
        host: str='', 
        port: int | None=None, 
        client_id: int | None=None, 
    ):
        super().__init__(env, Broker.IB, name=name)
        # FIXME: load from .env
        self._host = host
        self._port = port
        self._client_id = client_id
        assert host, f'`host` must be provided for {self.env} trading environment'
        assert port, f'`port` must be provided for {self.env} trading environment'
        assert client_id, f'`client_id` must be provided for {self.env} trading environment'
            