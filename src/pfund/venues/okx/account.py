from __future__ import annotations

from pfund.entities.accounts import APIKeyAccount
from pfund.enums import TradingVenue
from pfund_blocks.enums.env import Environment


class OKXAccount(APIKeyAccount):
    def __init__(
        self,
        env: Environment | str,
        name: str = "",
        key: str = "",
        secret: str = "",
    ):
        super().__init__(
            env=env, venue=TradingVenue.OKX, name=name, key=key, secret=secret
        )
