from __future__ import annotations

from typing import TYPE_CHECKING, Literal, override

if TYPE_CHECKING:
    from pfund_blocks.typing import AccountName, Currency, ProductName, TradingVenue
    from pfund_blocks.utils import DotenvStore

from decimal import Decimal

from pfund_blocks.entities import BaseAccount
from pfund_blocks.enums.env import Environment


class SimulatedAccount(BaseAccount):
    def __init__(
        self,
        venue: TradingVenue,
        name: AccountName = "",
        env: Literal[Environment.BACKTEST, Environment.SANDBOX, "BACKTEST", "SANDBOX"]
        | None = None,
        balances: dict[Currency, Decimal | int | float | str] | None = None,
        positions: dict[ProductName, Decimal | int | float | str] | None = None,
    ):
        """
        Args:
            venue: The trading venue being simulated, e.g. "BYBIT".
            name: The name of the account.
            env: The environment the account is in, only env=BACKTEST or env=SANDBOX.
                If None, it is set internally by the engine when the account is added.
            balances: Starting balance per currency, e.g. {"USDT": 10_000}.
            positions: Starting position per product name, e.g. {"BTC_USDT_PERP": 0.1}.
        """
        # NOTE: str(amount) so that floats keep their printed value, e.g. Decimal(0.1) != Decimal("0.1")
        self._balances: dict[Currency, Decimal] = {
            ccy.upper(): Decimal(str(amount))
            for ccy, amount in (balances or {}).items()
        }
        self._positions: dict[ProductName, Decimal] = {
            pdt.upper(): Decimal(str(amount))
            for pdt, amount in (positions or {}).items()
        }
        super().__init__(venue=venue, name=name, env=env)

    @override
    def _set_env(self, env: Environment | str):
        if self._env is not None:
            raise ValueError(f"{self!r} already has env={self._env}, cannot set {env=}")
        _env = Environment(env.upper())
        if not _env.is_simulated():
            raise ValueError(
                f"{self.__class__.__name__} only supports env=BACKTEST or env=SANDBOX, got {env=}"
            )
        self._env = _env

    # NOTE: a simulated account has no credentials
    @override
    def _load_credentials(self, dotenv: DotenvStore) -> None:
        pass

    @override
    def _assert_credentials_provided(self) -> None:
        pass

    @property
    def balances(self) -> dict[Currency, Decimal]:
        return self._balances

    @property
    def positions(self) -> dict[ProductName, Decimal]:
        return self._positions

    @override
    def __str__(self) -> str:
        return "[SIMULATED] " + super().__str__()

    @override
    def __repr__(self) -> str:
        return "[SIMULATED]" + super().__repr__()
