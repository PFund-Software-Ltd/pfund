from __future__ import annotations
from typing import Any, ClassVar

from pfund.enums import Environment, TradingVenue


class BaseAccount:
    _num: ClassVar[int] = 0

    @classmethod
    def _next_id(cls):
        cls._num += 1
        return str(cls._num)

    def _get_default_name(self):
        return f"{self.__class__.__name__}-{self._next_id()}"

    def _normalize_name(self, name: str) -> str:
        name = name or self._get_default_name()
        if "account" not in name.lower():
            name += "_account"
        return name

    def __init__(self, env: Environment, venue: TradingVenue, name: str = ""):
        self._env = Environment[env.upper()]
        self._venue = TradingVenue[venue.upper()]
        self.name: str = self._normalize_name(name)

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def venue(self) -> TradingVenue:
        return self._venue

    def __str__(self):
        return f"Env={self._env} | Venue={self._venue} | Account={self.name}"

    def __repr__(self):
        return ":".join([self._env, self._venue, self.name])

    def __eq__(self, other: Any):
        if not isinstance(other, BaseAccount):
            return NotImplemented
        return (
            self._env == other._env
            and self._venue == other._venue
            and self.name == other.name
        )

    def __hash__(self):
        return hash((self._env, self._venue, self.name))
