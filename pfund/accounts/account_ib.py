import os

from pfund._typing import tEnvironment
from pfund.accounts.account_base import BaseAccount
from pfund.enums import Environment


class IBAccount(BaseAccount):
    _default_client_id = 0
    
    @classmethod
    def _next_default_client_id(cls):
        cls._default_client_id += 1
        return cls._default_client_id
    
    def __init__(self, env: tEnvironment, name: str='', host: str='', port: int | None=None, client_id: int | None=None):
        '''
        Args:
            name: account code, e.g. DU123456 for paper trading, U123456 for live trading
        '''
        super().__init__(env=env, trading_venue='IB', name=name)
        # remove the added "_account" suffix
        if self.name.endswith('_account'):
            self.name = self.name.rsplit('_account', 1)[0]
        self._host = host or os.getenv(f'{self.tv}_HOST', '127.0.0.1')
        self._port = port or os.getenv(f'{self.tv}_PORT', None)
        if self._port:
            self._port = int(self._port)
        self._client_id = client_id or os.getenv(f'{self.tv}_CLIENT_ID', self._next_default_client_id())
        if self._client_id:
            self._client_id = int(self._client_id)
        if self._env in [Environment.SANDBOX, Environment.PAPER, Environment.LIVE]:
            assert self._host, f'{self.tv} host must be provided, please set `{self.tv}_HOST` in .env.{self._env.lower()} file, or in add_account(..., host=...).'
            assert self._port, f'''\033[93m
                {self.tv} port must be provided for, please set `{self.tv}_PORT` in .env.{self._env.lower()} file, or in add_account(..., port=...).
                You can find your default socket port in Trader Workstation (TWS):
                ⚙️ icon on the top right corner -> API -> Settings -> Socket port
                or
                You can find your default socket port in IB Gateway:
                Configure -> Settings -> API -> Settings -> Socket port\033[0m
            '''
            assert self._client_id, f'{self.tv} client id must be provided, please set `{self.tv}_CLIENT_ID` in .env.{self._env.lower()} file, or in add_account(..., client_id=...).'

    @property
    def host(self):
        return self._host
    
    @property
    def port(self):
        return self._port
    
    @property
    def client_id(self):
        return self._client_id