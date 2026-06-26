from decimal import Decimal
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, ConfigDict


class BaseMarket(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    base_asset: str
    quote_asset: str
    asset_type: str
    tick_size: Decimal | None = Field(
        default=None, description="Tick size, minimum price increment"
    )
    lot_size: Decimal | None = Field(
        default=None, description="Lot size, minimum quantity incrementc"
    )
    min_leverage: Decimal | None = Field(default=None, description="Minimum leverage")
    max_leverage: Decimal | None = Field(default=None, description="Maximum leverage")
    leverage_step: Decimal | None = Field(
        default=None, description="Leverage step size"
    )
    min_price: Decimal | None = Field(default=None, description="Minimum price")
    max_price: Decimal | None = Field(default=None, description="Maximum price")
    min_quantity: Decimal | None = Field(
        default=None, description="Minimum order quantity"
    )
    max_quantity: Decimal | None = Field(
        default=None, description="Maximum order quantity"
    )
    listed_at: datetime | None = Field(
        default=None, description="Date the market was listed"
    )
    expiration: datetime | None = Field(default=None, description="expiration date")
    settle_currency: str | None = Field(default=None, description="Settlement currency")
    funding_interval: timedelta | None = Field(
        default=None, description="Funding interval"
    )
