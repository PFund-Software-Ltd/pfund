from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.entities.accounts.account_base import BaseAccount
    from pfund.entities.orders.order_base import BaseOrder
    from pfund.entities.products.product_base import BaseProduct

import importlib
from enum import StrEnum
from pfund.enums.broker import Broker
from pfund.enums.crypto_exchange import CryptoExchange


class TradingVenue(StrEnum):
    IBKR = Broker.IBKR
    BYBIT = CryptoExchange.BYBIT
    
    @property
    def broker(self) -> Broker:
        if self.value in CryptoExchange.__members__:
            return Broker.CRYPTO
        elif self.value in Broker.__members__:
            return Broker[self.value]
        else:
            raise ValueError(f"No matching Broker for trading venue: {self.value}")

    @property
    def order_class(self) -> type[BaseOrder]:
        if self == TradingVenue.IBKR:
            class_name = f'{self}Order'
        else:
            class_name = f'{self.capitalize()}Order'
        Order = getattr(importlib.import_module(f'pfund.entities.orders.order_{self.lower()}'), class_name)
        return Order
    
    @property
    def product_class(self) -> type[BaseProduct]:
        if self == TradingVenue.IBKR:
            class_name = f'{self}Product'
        else:
            class_name = f'{self.capitalize()}Product'
        Product = getattr(importlib.import_module(f'pfund.entities.products.product_{self.lower()}'), class_name)
        return Product
        
    @property
    def account_class(self) -> type[BaseAccount]:
        if self == TradingVenue.IBKR:
            class_name = f'{self}Account'
        elif self.value in CryptoExchange.__members__:
            class_name = 'CryptoAccount'
            return getattr(importlib.import_module('pfund.entities.accounts.account_crypto'), class_name)
        else:
            class_name = f'{self.capitalize()}Account'
        Account = getattr(importlib.import_module(f'pfund.entities.accounts.account_{self.lower()}'), class_name)
        return Account
