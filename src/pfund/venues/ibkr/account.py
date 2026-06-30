from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.engines.engine_context import EngineContext

import warnings

from pfund.entities import BaseAccount
from pfund.enums import Environment, TradingVenue


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
        self._host: str = host
        self._port: int | None = port
        self._client_id: int | None = client_id

    def _load_env_vars_from_context(self, context: EngineContext):
        self._host = (
            self._host or context.get_env(f"{self.venue}_HOST") or self.DEFAULT_HOST
        )
        port = (
            self._port
            or context.get_env(f"{self.venue}_{self._env}_PORT")
            or self.DEFAULT_PORTS.get(self._env)
        )
        self._port = int(port) if port else None

        client_id = self._client_id or context.get_env(f"{self.venue}_CLIENT_ID")
        if client_id:
            self._client_id = int(client_id)
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
