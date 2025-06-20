from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.exchanges.exchange_base import BaseExchange

from enum import StrEnum


class CryptoExchange(StrEnum):
    BINANCE = 'BINANCE'
    BYBIT = 'BYBIT'
    OKX = 'OKX'

    @property
    def exchange_class(self) -> type[BaseExchange]:
        import importlib
        return getattr(importlib.import_module(f'pfund.exchanges.{self.lower()}.exchange'), 'Exchange')