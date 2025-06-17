from pfund.typing import tEnvironment, tCryptoExchange
from pfund.accounts.account_base import BaseAccount
from pfund.enums import CryptoExchange


class CryptoAccount(BaseAccount):
    def __init__(
        self, 
        env: tEnvironment,
        exchange: tCryptoExchange, 
        name: str='', 
        key: str='', 
        secret: str='',
    ):
        self.exchange = CryptoExchange[exchange.upper()]
        if self.exchange == CryptoExchange.BYBIT:
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
        return self.exchange
    
    def __str__(self):
        return f'Broker={self.broker}|Exchange={self.exchange}|Account={self.name}'

    def __repr__(self):
        return f'{self.broker}:{self.exchange}:{self.name}'
   
    def __eq__(self, other):
        if not isinstance(other, CryptoAccount):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self._env == other._env
            and self.broker == other.broker
            and self.exchange == other.exchange
            and self.name == other.name
        )
     
    def __hash__(self):
        return hash((self._env, self.broker, self.exchange, self.name))