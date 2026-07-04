from __future__ import annotations

from pfund.entities.accounts import APIKeyAccount
from pfund.enums import Environment, TradingVenue


class BybitAccount(APIKeyAccount):
    def __init__(
        self,
        env: Environment | str,
        name: str = "",
        key: str = "",
        secret: str = "",
    ):
        super().__init__(
            env=env, venue=TradingVenue.BYBIT, name=name, key=key, secret=secret
        )

    # NOTE: only supports unified accounts
    @property
    def type(self) -> str:
        return "UNIFIED"
