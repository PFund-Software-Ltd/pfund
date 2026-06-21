from __future__ import annotations
from typing import ClassVar, TypedDict

import time
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from pfund.entities import BaseProduct, Quantity
from pfund.enums import PositionMode, Side, TradingVenue


class PositionUpdate(TypedDict, total=False):
    size: Quantity
    avg_price: Decimal | None
    liq_price: Decimal | None
    unrealized_pnl: Decimal | None
    realized_pnl: Decimal | None


class PositionSnapshot(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    updated_at: float = Field(default_factory=time.time)
    size: Quantity = Field(default=Quantity(0))
    avg_price: Decimal | None = Field(
        default=None, ge=0, description="Average price of the position"
    )
    liq_price: Decimal | None = Field(
        default=None, ge=0, description="Liquidation price of the position"
    )
    unrealized_pnl: Decimal | None = Field(
        default=None, description="Unrealized PnL of the position"
    )
    realized_pnl: Decimal | None = Field(
        default=None, description="Realized PnL of the position"
    )


class BasePosition(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )
    _Snapshot: ClassVar[type[PositionSnapshot]] = PositionSnapshot
    _snapshot: PositionSnapshot = PrivateAttr(default_factory=PositionSnapshot)

    mode: PositionMode = Field(default=PositionMode.NORMAL)
    product: BaseProduct

    @property
    def venue(self) -> TradingVenue:
        assert self.product.venue is not None, "venue is None"
        return self.product.venue

    @computed_field
    @property
    def size(self) -> Quantity:
        return self._snapshot.size

    @property
    def side(self) -> Side:
        return Side(self.size)

    @property
    def quantity(self) -> Quantity:
        return Quantity(abs(self.size))

    qty = quantity

    @computed_field
    @property
    def avg_price(self) -> Decimal:
        assert self._snapshot.avg_price is not None, "avg_price is None"
        return self._snapshot.avg_price

    entry_price = entry_px = avg_px = avg_price

    @property
    def updated_at(self) -> float:
        return self._snapshot.updated_at

    def is_empty(self) -> bool:
        return self.size == 0

    @computed_field
    @property
    def unrealized_pnl(self) -> Decimal | None:
        return self._snapshot.unrealized_pnl

    upnl = unrealized_pnl

    def compute_unrealized_pnl(self, mark_price: Decimal) -> Decimal:
        if self.is_empty():
            return Decimal(0)
        assert self.avg_price is not None, (
            "Average price is None, cannot compute unrealized PnL"
        )
        return self.size * (mark_price - self.avg_price)

    @computed_field
    @property
    def realized_pnl(self) -> Decimal | None:
        return self._snapshot.realized_pnl

    rpnl = realized_pnl

    def on_snapshot_update(self, update: PositionUpdate) -> None:
        self._snapshot = self._Snapshot.model_validate(update)

    # TODO
    # def on_trade_update(self, order: BaseOrder):
    #     ttl_size = self.size + o.ltz
    #     new_side = sign(ttl_size)
    #     ttl_qty = abs(ttl_size)
    #     # calc avg_px
    #     if new_side != 0:
    #         if o.side != self.side:
    #             if o.ltq > self.qty:
    #                 avg_px = o.ltp
    #             else:
    #                 avg_px = self.long_avg_px if self.side == 1 else self.short_avg_px
    #         else:
    #             avg_px = self.long_avg_px if self.side == 1 else self.short_avg_px
    #             qty = self.long_qty if self.side == 1 else self.short_qty
    #             avg_px = (avg_px * qty + o.ltp * o.ltq) / ttl_qty
    #     else:
    #         avg_px = Decimal(0)
    #     update = {
    #         new_side: {
    #             "qty": ttl_qty,
    #             "avg_px": avg_px,
    #             "liquidation_px": Decimal(0),
    #             "unrealized_pnl": Decimal(0),
    #             "realized_pnl": Decimal(0),
    #         }
    #     }
    #     ts = o.ltt
    #     self.logger.debug(
    #         f"update {self.exch} {self.pdt} {update} by oid {o.oid} ({o.lts}@{o.ltp}) ts={ts}"
    #     )
    #     self.on_update(update)

    def __str__(self):
        return (
            f"Venue={self.venue} | Product={self.product}\n"
            f"Size={self.size} | AveragePrice={self._snapshot.avg_price} | LiquidationPrice={self._snapshot.liq_price}\n"
            f"UnrealizedPnL={self.unrealized_pnl} | RealizedPnL={self.realized_pnl}"
        )

    def __repr__(self):
        return (
            f"{self.venue} | {self.product}\n"
            f"{self.size}@{self._snapshot.avg_price} (liquidate@{self._snapshot.liq_price})\n"
            f"upnl={self.unrealized_pnl} | rpnl={self.realized_pnl}"
        )
