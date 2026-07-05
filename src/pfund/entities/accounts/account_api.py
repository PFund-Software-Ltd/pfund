from __future__ import annotations

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
        self._key: str = key or (self._dotenv.get(f"{self.venue}_API_KEY", "") or "")
        self._secret: str = secret or (
            self._dotenv.get(f"{self.venue}_API_SECRET", "") or ""
        )
        if self._env in [Environment.PAPER, Environment.LIVE]:
            if not self._key or not self._secret:
                raise ValueError(
                    f"{self.venue} API KEY and SECRET must be provided, \n"
                    + f"please set `{self.venue}_API_KEY` and `{self.venue}_API_SECRET` in .env.{self._env.lower()} file,\n"
                    + "or by strategy.add_account(..., key=..., secret=...)"
                )

    def _assert_no_credential_leak_in_simulated_env(self):
        if self._key or self._secret:
            raise ValueError(
                f"{self.venue} API KEY and SECRET should NOT be provided for {self._env} environment, \n"
                + f"please remove them in .env.{self._env.lower()} file (if any),\n"
                + "or remove key= and secret= from strategy.add_account()"
            )

    @property
    def key(self):
        return self._key

    @property
    def secret(self):
        return self._secret
