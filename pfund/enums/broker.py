from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.brokers.broker_base import BaseBroker
    from pfund.accounts.account_base import BaseAccount

from enum import StrEnum


class Broker(StrEnum):
    IB = 'IB'
    CRYPTO = 'CRYPTO'
    DAPP = 'DAPP'

    @property
    def broker_class(self) -> type[BaseBroker]:
        """Returns the corresponding Broker class for this broker."""
        import pfund as pf
        broker_name = {
            Broker.IB: 'IBBroker',
            Broker.CRYPTO: 'CryptoBroker',
            Broker.DAPP: 'DappBroker',
        }[self]
        return getattr(pf, broker_name)
     