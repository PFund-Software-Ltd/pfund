from pfund.accounts.account_base import BaseAccount
from pfund.enums import Environment, Broker, CryptoExchange


class CryptoAccount(BaseAccount):
    def __init__(self, env: Environment, exch: str, key='', secret='', name='', **kwargs):
        self.exch = CryptoExchange[exch.upper()]
        if self.exch == CryptoExchange.BYBIT:
            from pfund import print_warning
            print_warning('account type is "UNIFIED" for Bybit by default, please make sure it is set correctly on Bybit\'s website')
        super().__init__(env, Broker.CRYPTO, name=name, key=key, secret=secret, **kwargs)
        if self.env in [Environment.PAPER, Environment.LIVE]:
            assert key, f'API `key` must be provided for {self.env} trading environment'
            assert secret, f'API `secret` must be provided for {self.env} trading environment'

    def __str__(self):
        return f'Broker={self.bkr.value}|Exchange={self.exch.value}|Account={self.name}|Strategy={self.strat}'

    def __repr__(self):
        return f'{self.bkr.value}:{self.exch.value}:{self.name}'
   
    def __eq__(self, other):
        if not isinstance(other, CryptoAccount):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self.env == other.env
            and self.bkr == other.bkr
            and self.exch == other.exch
            and self.name == other.name
        )
     
    def __hash__(self):
        return hash((self.env, self.bkr, self.exch, self.name))