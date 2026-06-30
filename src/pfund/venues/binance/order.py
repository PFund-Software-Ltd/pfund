from __future__ import annotations
from typing import ClassVar

from decimal import Decimal
import builtins

from pydantic import Field, model_validator

from pfund.entities.orders.trailing_stop import TrailingStop
from pfund.entities import BaseOrder


class BinanceTrailingStop(TrailingStop):
    activation_price: Decimal | None = Field(
        default=None,
        description="""
        market price at which the trailing stop becomes active and begins
        trailing; until it is reached the stop is dormant and does not move.
        if omitted, trailing starts immediately from the current market price.
        """,
    )

    @model_validator(mode="after")
    def _validate_binance_trailing_stop(self) -> BinanceTrailingStop:
        if self.percent is None:
            raise ValueError(
                "Binance trailing stop requires percent (callbackRate); "
                + "TRAILING_STOP_MARKET trails by rate only and does not accept amount"
            )
        if self.limit_offset is not None:
            raise ValueError(
                "Binance trailing stop is market-only (TRAILING_STOP_MARKET); "
                + "limit_offset is not supported"
            )
        return self


class BinanceOrder(BaseOrder):
    # NOTE: use builtins.type to avoid collision with property "type"
    TrailingStop: ClassVar[builtins.type[BinanceTrailingStop]] = BinanceTrailingStop
    SUBMITTED_AS_PENDING: ClassVar[bool] = True

    trailing_stop: BinanceTrailingStop | None = Field(default=None)
