from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.brokers.broker_base import BaseBroker

from enum import StrEnum


class Broker(StrEnum):
    IBKR = 'IBKR'
    CRYPTO = 'CRYPTO'
    DEFI = 'DEFI'

    @property
    def broker_class(self) -> type[BaseBroker]:
        """Returns the corresponding Broker class for this broker."""
        import pfund as pf
        broker_name = {
            Broker.IBKR: 'IBKR',
            Broker.CRYPTO: 'CryptoBroker',
            Broker.DEFI: 'DeFiBroker',
        }[self]
        return getattr(pf, broker_name)
     