from typing import ClassVar

from pfund.enums import Environment, Broker
from pfund.typing import tEnvironment, tBroker


class BaseAccount:
    _num: ClassVar[int] = 0

    @classmethod
    def _next_account_id(cls):
        cls._num += 1
        return str(cls._num)
    
    def _get_default_name(self):
        return f"{self.__class__.__name__}-{self._next_account_id()}"
    
    def __init__(self, env: tEnvironment, bkr: tBroker, name: str):
        self._env = Environment[env.upper()]
        self._bkr = Broker[bkr.upper()]
        self.name = name or self._get_default_name()
        
    @property
    def bkr(self) -> Broker:
        return self._bkr
    
    def __str__(self):
        return f'Broker={self._bkr.value}|Account={self.name}'

    def __repr__(self):
        return f'{self._bkr.value}:{self.name}'

    def __eq__(self, other):
        if not isinstance(other, BaseAccount):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self._env == other._env
            and self._bkr == other._bkr
            and self.name == other._name
        )
        
    def __hash__(self):
        return hash((self._env, self._bkr, self.name))