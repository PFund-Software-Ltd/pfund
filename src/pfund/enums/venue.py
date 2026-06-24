from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.venues.venue_base import BaseVenue

from enum import StrEnum


class TradingVenue(StrEnum):
    IBKR = "IBKR"
    ALPACA = "ALPACA"
    HYPERLIQUID = "HYPERLIQUID"
    BINANCE = "BINANCE"
    BYBIT = "BYBIT"
    OKX = "OKX"

    @property
    def venue_class(self) -> type[BaseVenue]:
        if self == TradingVenue.IBKR:
            from pfund.venues.ibkr.venue import InteractiveBrokers

            return InteractiveBrokers
        elif self == TradingVenue.BYBIT:
            from pfund.venues.bybit.venue import Bybit

            return Bybit
        else:
            raise ValueError(f"Unknown venue: {self}")
