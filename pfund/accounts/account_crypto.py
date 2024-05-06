from pfund.accounts.account_base import BaseAccount
from pfund.const.common import SUPPORTED_BYBIT_ACCOUNT_TYPES


class CryptoAccount(BaseAccount):
    def __init__(self, env, exch, key='', secret='', acc='', **kwargs):
        self.exch = exch.upper()
        if self.exch == 'BYBIT':
            assert 'account_type' in kwargs and kwargs['account_type'].upper() in SUPPORTED_BYBIT_ACCOUNT_TYPES, \
            f"kwarg 'account_type' must be provided for exchange {self.exch}, {SUPPORTED_BYBIT_ACCOUNT_TYPES=}"
        super().__init__(env, 'CRYPTO', acc=acc, key=key, secret=secret, **kwargs)
        if self.env in ['PAPER', 'LIVE']:
            assert key, f'API `key` must be provided for {self.env} trading environment'
            assert secret, f'API `secret` must be provided for {self.env} trading environment'

    def __str__(self):
        return f'Broker={self.bkr}|Exchange={self.exch}|Account={self.name}|Strategy={self.strat}'

    def __repr__(self):
        return f'{self.bkr}-{self.exch}-{self.name}'
   
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