from __future__ import annotations
from typing import Literal, Any

from decimal import Decimal

from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator

from pfund.typing import Currency
from pfund.entities.trades.quantity import Quantity
from pfund.enums import Side


class Trade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: float
    trade_id: str
    order_id: str = Field(description="which order this trade belongs to")
    side: Side | Literal["BUY", "SELL", 1, -1]
    price: Decimal = Field(gt=0.0, description="traded price")
    quantity: Quantity = Field(gt=0.0, description="traded quantity")
    fee: Decimal | None = Field(
        default=None,
        description="""
        the fee exactly as charged by the trading venue, in its native currency
        (e.g. 0.0001 BTC). Kept as-is for reconciliation against the venue's records
        """,
    )
    fee_currency: Currency = Field(
        default="",
        description="the native currency the fee was charged in (e.g. BTC), pairs with `fee`",
    )
    booked_fee: Decimal | None = Field(
        default=None,
        description="""
        the value of `fee` converted into the fund's booking (reporting) currency,
        frozen at trade time using the price at this trade's timestamp. This is a
        static historical cost: it must NOT be revalued later (e.g. with an EOD price),
        otherwise the booking currency's price fluctuation would leak into the fee.
        Use this for aggregating total fees across trades/products in one common unit.
        """,
    )
    booked_fee_currency: Currency = Field(default="USD")
    is_maker: bool | None = Field(default=None)

    @field_validator("side", mode="before")
    @classmethod
    def _validate_side(cls, value: Any) -> Side:
        if isinstance(value, str):
            side = Side[value.upper()]
        elif isinstance(value, int):
            side = Side(value)
        else:
            raise ValueError(f"Invalid side: {value}")
        return side

    @computed_field
    @property
    def size(self) -> Quantity:
        return self.quantity * Side(self.side)

    @property
    def qty(self) -> Quantity:
        return self.quantity

    @property
    def px(self) -> Decimal:
        return self.price

    def __str__(self) -> str:
        return (
            f"TradeID={self.trade_id} | OrderID={self.order_id} | Timestamp={self.timestamp}\n"
            f"Side={Side(self.side).name} | Price={self.price} | Quantity={self.quantity}\n"
            f"Fee={self.fee} | FeeCurrency={self.fee_currency} | IsMaker={self.is_maker}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.trade_id} | order={self.order_id}\n"
            f"{self.size}@{self.price} | fee={self.fee} {self.fee_currency} | maker={self.is_maker}"
        )
