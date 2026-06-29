from __future__ import annotations
from typing import ClassVar

from decimal import Decimal

from pydantic import Field, model_validator

from pfund.entities.orders.trailing_stop import TrailingStop
from pfund.entities import BaseOrder


class BybitTrailingStop(TrailingStop):
    activation_price: Decimal | None = Field(
        default=None,
        description="""
        market price at which the trailing stop becomes active and begins
        trailing; until it is reached the stop is dormant and does not move.
        if omitted, trailing starts immediately from the current market price.
        """,
    )

    @model_validator(mode="after")
    def _validate_bybit_trailing_stop(self) -> BybitTrailingStop:
        if self.amount is None:
            raise ValueError(
                "Bybit trailing stop requires amount (absolute price distance); "
                + "the V5 API trails by distance only and does not accept percent"
            )
        if self.limit_offset is not None:
            raise ValueError(
                "Bybit trailing stop has no limit variant; "
                + "it reduces at market on trigger, so limit_offset is not supported"
            )
        return self


class BybitOrder(BaseOrder):
    TrailingStop: ClassVar[type[BybitTrailingStop]] = BybitTrailingStop
    SUBMITTED_AS_PENDING: ClassVar[bool] = True

    trailing_stop: BybitTrailingStop | None = Field(default=None)
