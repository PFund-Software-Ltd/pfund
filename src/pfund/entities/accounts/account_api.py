from __future__ import annotations
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pfund.engines.engine_context import EngineContext

from pfund.entities.accounts.account_base import BaseAccount
from pfund.enums import Environment, TradingVenue


class APIKeyAccount(BaseAccount):
    def __init__(
        self,
        env: Environment | str,
        venue: TradingVenue | str,
        name: str = "",
        key: str = "",
        secret: str = "",
    ):
        super().__init__(env=env, venue=venue, name=name)
        self._key: str = key
        self._secret: str = secret

    def _load_env_vars_from_context(self, context: EngineContext):
        self._key = self._key or cast(str, context.get_env(f"{self.venue}_API_KEY", ""))
        self._secret = self._secret or cast(
            str, context.get_env(f"{self.venue}_API_SECRET", "")
        )
        if self._env in [Environment.PAPER, Environment.LIVE]:
            if not self._key or not self._secret:
                raise ValueError(
                    f"{self.venue} API KEY and SECRET must be provided, \n"
                    + f"please set `{self.venue}_API_KEY` and `{self.venue}_API_SECRET` in .env.{self._env.lower()} file,\n"
                    + "or by strategy.add_account(..., key=..., secret=...)"
                )

    @property
    def key(self):
        return self._key

    @property
    def secret(self):
        return self._secret
