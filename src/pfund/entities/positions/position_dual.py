from __future__ import annotations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from decimal import Decimal
    from pfund.entities.positions.position_base import PositionUpdate

from pydantic import PrivateAttr, Field, model_validator

from pfund.entities.positions.position_base import BasePosition, PositionSnapshot
from pfund.enums import Side, PositionMode


class DualPosition(BasePosition):
    mode: PositionMode = Field(default=PositionMode.DUAL)
    _long_position: BasePosition | None = PrivateAttr(default=None)
    _short_position: BasePosition | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _init_legs(self):
        # each leg is a normal single-sided position; only the container is DUAL
        self._long_position = BasePosition(product=self.product)
        self._short_position = BasePosition(product=self.product)
        return self

    @property
    def long(self) -> BasePosition:
        assert self._long_position is not None
        return self._long_position

    @property
    def short(self) -> BasePosition:
        assert self._short_position is not None
        return self._short_position

    def is_empty(self) -> bool:
        return self.long.is_empty() and self.short.is_empty()

    def on_update(self, side: Side, update: PositionUpdate) -> None:
        # route to the per-side leg, then derive the net snapshot
        if side is Side.LONG:
            self.long.on_update(update)
        elif side is Side.SHORT:
            self.short.on_update(update)
        else:
            raise ValueError(
                f"{self.mode} mode requires a side (LONG/SHORT), got {side!r}"
            )
        self._combine()

    def _combine(self) -> None:
        """Derives the net snapshot from the long and short legs (DUAL mode)."""
        net_size = self.long.size + self.short.size
        if net_size == 0:
            avg_price = None
        else:
            avg_price = self.long.avg_price if net_size > 0 else self.short.avg_price
        self._snapshot = PositionSnapshot(
            size=net_size,
            avg_price=avg_price,
            liquidation_price=None,  # each side keeps its own; net liquidation is undefined in DUAL mode
            unrealized_pnl=self._combine_pnl("unrealized_pnl"),
            realized_pnl=self._combine_pnl("realized_pnl"),
        )

    def _combine_pnl(
        self, attr: Literal["unrealized_pnl", "realized_pnl"]
    ) -> Decimal | None:
        long_pnl = getattr(self.long, attr)
        short_pnl = getattr(self.short, attr)
        if long_pnl is None or short_pnl is None:
            return None
        return long_pnl + short_pnl

    def __str__(self):
        return (
            f"Venue={self.venue} | Product={self.product} | Mode={self.mode}\n"
            f"Net: Size={self.size} | AveragePrice={self.avg_price}\n"
            f"UnrealizedPnL={self.unrealized_pnl} | RealizedPnL={self.realized_pnl}\n"
            f"  Long: Quantity={self.long.quantity}@{self.long.avg_price} | upnl={self.long.unrealized_pnl} | rpnl={self.long.realized_pnl}\n"
            f"  Short: Quantity={self.short.quantity}@{self.short.avg_price} | upnl={self.short.unrealized_pnl} | rpnl={self.short.realized_pnl}"
        )

    def __repr__(self):
        return (
            f"{self.venue} | {self.product} | {self.mode}\n"
            f"net {self.size}@{self.avg_price}\n"
            f"upnl={self.unrealized_pnl} | rpnl={self.realized_pnl}\n"
            f"long {self.long.quantity}@{self.long.avg_price}\n"
            f"short {self.short.quantity}@{self.short.avg_price}"
        )
