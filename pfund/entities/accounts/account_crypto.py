import os

from pfund.entities.accounts.account_base import BaseAccount
from pfund.enums import Environment, CryptoExchange


class CryptoAccount(BaseAccount):
    def __init__(self, env: Environment | str, exchange: CryptoExchange | str, name: str='', key: str='', secret: str=''):
        super().__init__(env=env, trading_venue=exchange, name=name)
        self._key = key or os.getenv(f'{self.tv}_API_KEY', '')
        self._secret = secret or os.getenv(f'{self.tv}_API_SECRET', '')
        if self._env in [Environment.PAPER, Environment.LIVE]:
            assert self._key, f'{self.tv} API key must be provided, please set `{self.tv}_API_KEY` in .env.{self._env.lower()} file, or in add_account(..., key=...).'
            assert self._secret, f'{self.tv} API secret must be provided, please set `{self.tv}_API_SECRET` in .env.{self._env.lower()} file, or in add_account(..., secret=...).'

    @property
    def key(self):
        return self._key
    
    @property
    def secret(self):
        return self._secret
    
    def to_dict(self):
        return {
            **super().to_dict(),
            'key': self._key,
            'secret': self._secret,
        }