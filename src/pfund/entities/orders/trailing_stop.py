from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field, model_validator


class TrailingStop(BaseModel):
    amount: Decimal | None = Field(
        default=None,
        gt=0,
        description="absolute price offset the stop trails the market by",
    )
    percent: Decimal | None = Field(
        default=None,
        gt=0,
        description="relative offset (e.g. 1.5 == 1.5%) the stop trails the market by",
    )
    limit_offset: Decimal | None = Field(
        default=None,
        gt=0,
        description="""
        STOP_LIMIT only: gap from trigger to the limit price on activation;
        limit price = trigger -/+ limit_offset (sign from the order side).
        """,
    )

    @model_validator(mode="after")
    def _validate_offset(self) -> TrailingStop:
        if (self.amount is None) == (self.percent is None):
            raise ValueError("TrailingStop requires exactly one of amount or percent")
        return self
