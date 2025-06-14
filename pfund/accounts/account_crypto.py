from pfund.typing import tEnvironment, tCryptoExchange
from pfund.accounts.account_base import BaseAccount
from pfund.enums import CryptoExchange


class CryptoAccount(BaseAccount):
    def __init__(
        self, 
        env: tEnvironment,
        exch: tCryptoExchange, 
        name: str='', 
        key: str='', 
        secret: str='',
    ):
        self._exch = CryptoExchange[exch.upper()]
        if self._exch == CryptoExchange.BYBIT:
            from pfund import print_warning
            print_warning('account type is "UNIFIED" for Bybit by default, please make sure it is set correctly on Bybit\'s website')
        super().__init__(env=env, bkr='CRYPTO', name=name)
        # TODO: load from .env file, use .env.paper and .env.live?
        self._key = key
        self._secret = secret
        assert key, f'API `key` must be provided for {self._env} trading environment'
        assert secret, f'API `secret` must be provided for {self._env} trading environment'
        
    @property
    def exch(self) -> CryptoExchange:
        return self._exch
    
    def __str__(self):
        return f'Broker={self._bkr}|Exchange={self._exch}|Account={self.name}'

    def __repr__(self):
        return f'{self._bkr}:{self._exch}:{self.name}'
   
    def __eq__(self, other):
        if not isinstance(other, CryptoAccount):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self._env == other._env
            and self._bkr == other._bkr
            and self._exch == other._exch
            and self.name == other.name
        )
     
    def __hash__(self):
        return hash((self._env, self._bkr, self._exch, self.name))