from typing import ClassVar

import os

from pfund.enums import Environment, TradingVenue
from pfund.typing import tEnvironment, tTradingVenue


class BaseAccount:
    _num: ClassVar[int] = 0

    @classmethod
    def _next_account_id(cls):
        cls._num += 1
        return str(cls._num)
    
    def _get_default_name(self):
        return f"{self.__class__.__name__}-{self._next_account_id()}"
    
    def __init__(
        self, 
        env: tEnvironment, 
        trading_venue: tTradingVenue, 
        name: str='',
        key: str='',
        secret: str='',
    ):
        self._env = Environment[env.upper()]
        self.trading_venue = TradingVenue[trading_venue.upper()]
        self.name = name or self._get_default_name()
        if 'account' not in self.name.lower():
            self.name += "_account"
        self._key = key or os.getenv(f'{self.tv}_API_KEY', '')
        self._secret = secret or os.getenv(f'{self.tv}_API_SECRET', '')
        
    @property
    def tv(self) -> TradingVenue:
        return self.trading_venue
    
    def __str__(self):
        return f'TradingVenue={self.trading_venue}|Account={self.name}'

    def __repr__(self):
        return f'{self.trading_venue}:{self.name}'

    def __eq__(self, other):
        if not isinstance(other, BaseAccount):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self._env == other._env
            and self.trading_venue == other.trading_venue
            and self.name == other.name
        )
        
    def __hash__(self):
        return hash((self._env, self.trading_venue, self.name))