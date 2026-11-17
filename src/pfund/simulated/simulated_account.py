from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

if TYPE_CHECKING:
    from pfund_blocks.typing import AccountName, Currency, ProductName, TradingVenue

from decimal import Decimal

from pfund_blocks.enums.env import Environment


# DRAFT: an account on the simulated venue (the fake exchange), only allowed in env=BACKTEST or env=SANDBOX.
# it is a dumb carrier of the STARTING state only, the simulated venue seeds its bookkeeping from it when the account is added,
# and from then on the simulated venue (not this account) tracks the current balances and positions.
# TODO: decide whether to subclass the venue's account class or hold it (composition, like SimulatedVenue)
class SimulatedAccount:
    _num: ClassVar[int] = 0

    @classmethod
    def _next_id(cls) -> str:
        cls._num += 1
        return str(cls._num)

    def _get_default_name(self) -> str:
        return f"{self.__class__.__name__}-{self._next_id()}"

    def __init__(
        self,
        venue: TradingVenue,
        name: AccountName = "",
        env: Literal[Environment.BACKTEST, Environment.SANDBOX, "BACKTEST", "SANDBOX"] | None = None,
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
        self._env: Literal[Environment.BACKTEST, Environment.SANDBOX] | None = None
        self._venue: TradingVenue = venue.upper()
        self._name: AccountName = name or self._get_default_name()
        # NOTE: str(amount) so that floats keep their printed value, e.g. Decimal(0.1) != Decimal("0.1")
        self._balances: dict[Currency, Decimal] = {
            ccy.upper(): Decimal(str(amount)) for ccy, amount in (balances or {}).items()
        }
        self._positions: dict[ProductName, Decimal] = {
            pdt.upper(): Decimal(str(amount)) for pdt, amount in (positions or {}).items()
        }
        if env is not None:
            self._set_env(env)

    def _set_env(self, env: Literal[Environment.BACKTEST, Environment.SANDBOX, "BACKTEST", "SANDBOX"]):
        if self._env is not None:
            raise ValueError(f"{self!r} already has env={self._env}, cannot set {env=}")
        _env = Environment(env.upper())
        if not _env.is_simulated():
            raise ValueError(
                f"{self.__class__.__name__} only supports env=BACKTEST or env=SANDBOX, got {env=}"
            )
        self._env = cast("Literal[Environment.BACKTEST, Environment.SANDBOX]", _env)

    @property
    def env(self) -> Literal[Environment.BACKTEST, Environment.SANDBOX]:
        assert self._env is not None, "env is not set"
        return self._env

    @property
    def venue(self) -> TradingVenue:
        return self._venue

    @property
    def name(self) -> AccountName:
        return self._name

    @property
    def balances(self) -> dict[Currency, Decimal]:
        return self._balances

    @property
    def positions(self) -> dict[ProductName, Decimal]:
        return self._positions

    def __str__(self) -> str:
        return f"[SIMULATED] Env={self.env} | Venue={self._venue} | Account={self._name}"

    def __repr__(self) -> str:
        return "[SIMULATED]" + ":".join([self.env, self._venue, self._name])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SimulatedAccount):
            return NotImplemented
        return (
            self.env == other.env
            and self._venue == other._venue
            and self._name == other._name
        )

    def __hash__(self) -> int:
        return hash((self.env, self._venue, self._name))
