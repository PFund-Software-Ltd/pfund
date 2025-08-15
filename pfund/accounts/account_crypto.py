import os

from pfund._typing import tEnvironment, tCryptoExchange
from pfund.accounts.account_base import BaseAccount
from pfund.enums import Environment, TradingVenue


class CryptoAccount(BaseAccount):
    def __init__(self, env: tEnvironment, exchange: tCryptoExchange, name: str='', key: str='', secret: str=''):
        super().__init__(env=env, trading_venue=exchange, name=name, key=key, secret=secret)
        if self.tv == TradingVenue.BYBIT:
            from pfund import print_warning
            print_warning('account type is "UNIFIED" for Bybit by default, please make sure it is set correctly on Bybit\'s website')
        self._key = key or os.getenv(f'{self.tv}_API_KEY', '')
        self._secret = secret or os.getenv(f'{self.tv}_API_SECRET', '')
        if self._env in [Environment.SANDBOX, Environment.PAPER, Environment.LIVE]:
            assert self._key, f'{self.tv} API key must be provided, please set `{self.tv}_API_KEY` in .env.{self._env.lower()} file, or in add_account(..., key=...).'
            assert self._secret, f'{self.tv} API secret must be provided, please set `{self.tv}_API_SECRET` in .env.{self._env.lower()} file, or in add_account(..., secret=...).'

    @property
    def key(self):
        return self._key
    
    @property
    def secret(self):
        return self._secret