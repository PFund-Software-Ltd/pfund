import os

from pfund.entities.accounts.account_base import BaseAccount
from pfund.enums import Environment, TradingVenue


# TODO: for FUTURE integration with the US brokers — Tradier, TradeStation, Schwab, Tradovate.
class OAuthAccount(BaseAccount):
    """Account for venues that authenticate with an OAuth2 bearer token.

    Unlike HMAC venues, brokers such as Tradier don't sign each request: they send
    ``Authorization: Bearer <access_token>``. The hard part is the OAuth2 token lifecycle
    (obtaining/refreshing the access token) which is stateful and lives outside the signer;
    this account only carries the already-acquired token.
    """

    def __init__(
        self,
        env: Environment,
        venue: TradingVenue,
        name: str = "",
        token: str = "",
    ):
        super().__init__(env=env, venue=venue, name=name)
        self._token = token or os.getenv(f"{self.venue}_ACCESS_TOKEN", "")
        if self._env in [Environment.PAPER, Environment.LIVE]:
            if not self._token:
                raise ValueError(
                    f"{self.venue} ACCESS TOKEN must be provided, \n"
                    + f"please set `{self.venue}_ACCESS_TOKEN` in .env.{self._env.lower()} file,\n"
                    + "or by strategy.add_account(..., token=...)"
                )

    @property
    def token(self):
        return self._token
