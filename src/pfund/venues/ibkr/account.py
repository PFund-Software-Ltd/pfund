from __future__ import annotations
from typing import ClassVar

import warnings

from pfund.entities import BaseAccount
from pfund.enums import Environment, TradingVenue
from pfund.utils import DotenvStore


class InteractiveBrokersAccount(BaseAccount):
    _default_client_id: ClassVar[int] = 0
    DEFAULT_HOST: ClassVar[str] = "127.0.0.1"
    # default host-side ports, mirrors `IBKR_LIVE_PORT`/`IBKR_PAPER_PORT` in compose.yml
    DEFAULT_PORTS: ClassVar[dict[Environment, int]] = {
        Environment.LIVE: 4001,
        Environment.PAPER: 4002,
    }

    def _get_default_name(self):
        default_name = super()._get_default_name().replace("InteractiveBrokers", "IBKR")
        return default_name

    @classmethod
    def _next_default_client_id(cls) -> int:
        cls._default_client_id += 1
        return cls._default_client_id

    def __init__(
        self,
        env: Environment | str,
        name: str = "",
        host: str = "",
        port: int | None = None,
        client_id: int | None = None,
    ):
        """
        Args:
            name: account code, e.g. DU123456 for paper trading, U123456 for live trading
        """
        super().__init__(env=env, venue=TradingVenue.IBKR, name=name)
        dotenv = DotenvStore(env=self._env)
        self._host: str = host or dotenv.get(f"{self.venue}_HOST") or self.DEFAULT_HOST
        self._port = (
            port
            or dotenv.get(f"{self.venue}_{self._env}_PORT")
            or self.DEFAULT_PORTS.get(self._env)
        )
        if self._port:
            self._port = int(self._port)

        self._client_id = client_id or dotenv.get(f"{self.venue}_CLIENT_ID")
        if self._client_id:
            self._client_id = int(self._client_id)
        else:
            self._client_id = self._next_default_client_id()
            warnings.warn(
                f"{self.venue} client_id not set, auto-assigned {self._client_id}\n"
                + f"set env var `{self.venue}_CLIENT_ID` or strategy.add_account(..., client_id=...) to assign it",
                category=UserWarning,
                stacklevel=2,
            )

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    @property
    def client_id(self):
        return self._client_id
