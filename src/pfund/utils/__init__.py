from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple, Literal

from decimal import Decimal

from pfund.utils.zmq_graph import show_zmq_graph

if TYPE_CHECKING:
    from pfund.enums import Environment


__all__ = [
    "Timestamp",
    "DotenvStore",
    "trim_trailing_zeros",
    "show_zmq_graph",
]


# used to tell if a timestamp is from the venue or created by pfund internally
class Timestamp(NamedTuple):
    value: float
    source: Literal["venue", "pfund"]


class DotenvStore:
    """Read-only view over a single environment's `.env.{env}` file.

    Vars are loaded once at construction and kept in a private dict; they never
    enter `os.environ`, so two engines with different envs in the same process
    can't collide — no prefix is needed.
    """

    def __init__(self, env: Environment | str):
        from pfund.enums import Environment

        self._env = Environment[env.upper()]
        self._vars: dict[str, str] = self._load(self._env)

    @staticmethod
    def _load(env: Environment) -> dict[str, str]:
        from dotenv import dotenv_values, find_dotenv

        # load env vars manually to avoid loading into os.environ
        # NOTE: this allows multiple engines with different envs in the same process
        env_file_path = find_dotenv(
            filename=f".env.{env.lower()}", usecwd=True, raise_error_if_not_found=False
        )
        return dict(dotenv_values(env_file_path))  # pyright: ignore[reportReturnType]

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._vars.get(key, default)


def trim_trailing_zeros(value: Decimal) -> Decimal:
    """Trims unnecessary trailing zeros from a Decimal without changing its value.

    e.g. 0.010 -> 0.01, 25000.00 -> 25000

    For whole numbers, `to_integral()` is used instead of `normalize()` to avoid
    the exponent form normalize() produces for integers (e.g. 25000 -> 2.5E+4).
    """
    integral = value.to_integral()
    return integral if value == integral else value.normalize()
