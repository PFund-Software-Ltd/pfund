import os

from pfund.entities.accounts.account_base import BaseAccount
from pfund.enums import Environment, TradingVenue


class APIKeyAccount(BaseAccount):
    def __init__(
        self,
        env: Environment,
        venue: TradingVenue,
        name: str = "",
        key: str = "",
        secret: str = "",
    ):
        super().__init__(env=env, venue=venue, name=name)
        self._key = key or os.getenv(f"{self.venue}_API_KEY", "")
        self._secret = secret or os.getenv(f"{self.venue}_API_SECRET", "")
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
