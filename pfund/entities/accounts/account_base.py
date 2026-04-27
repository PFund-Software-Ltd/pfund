from typing import ClassVar, Any

from pfund.enums import Environment, TradingVenue


class BaseAccount:
    _num: ClassVar[int] = 0

    @classmethod
    def _next_account_id(cls):
        cls._num += 1
        return str(cls._num)
    
    def _get_default_name(self):
        return f"{self.__class__.__name__}-{self._next_account_id()}"
    
    def __init__(self, env: Environment | str, venue: TradingVenue | str, name: str=''):
        self._env = Environment[env.upper()]
        self.venue = TradingVenue[venue.upper()]
        self.name: str = name or self._get_default_name()
        if 'account' not in self.name.lower():
            self.name += "_account"
    
    def to_dict(self):
        return {
            'venue': self.venue,
            'name': self.name,
        }
        
    def __str__(self):
        return f'TradingVenue={self.venue}|Account={self.name}'

    def __repr__(self):
        return f'{self.venue}:{self.name}'

    def __eq__(self, other: Any):
        if not isinstance(other, BaseAccount):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self._env == other._env
            and self.venue == other.venue
            and self.name == other.name
        )
        
    def __hash__(self):
        return hash((self._env, self.venue, self.name))