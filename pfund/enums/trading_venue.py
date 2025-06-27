from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.accounts.account_base import BaseAccount
    from pfund.orders.order_base import BaseOrder
    from pfund.products.product_base import BaseProduct
    from pfund.enums import Broker

import importlib
from enum import StrEnum


class TradingVenue(StrEnum):
    IB = 'IB'
    BYBIT = 'BYBIT'
    
    @property
    def broker(self) -> Broker:
        from pfund.enums import Broker, CryptoExchange
        if self.value in CryptoExchange.__members__:
            return Broker.CRYPTO
        elif self.value in Broker.__members__:
            return Broker[self.value]
        else:
            raise ValueError(f"No matching Broker for trading venue: {self.value}")

    @property
    def order_class(self) -> type[BaseOrder]:
        if self == TradingVenue.IB:
            class_name = f'{self}Order'
        else:
            class_name = f'{self.capitalize()}Order'
        Order = getattr(importlib.import_module(f'pfund.orders.order_{self.lower()}'), class_name)
        return Order
    
    @property
    def product_class(self) -> type[BaseProduct]:
        if self == TradingVenue.IB:
            class_name = f'{self}Product'
        else:
            class_name = f'{self.capitalize()}Product'
        Product = getattr(importlib.import_module(f'pfund.products.product_{self.lower()}'), class_name)
        return Product
        
    @property
    def account_class(self) -> type[BaseAccount]:
        from pfund.enums import CryptoExchange
        if self == TradingVenue.IB:
            class_name = f'{self}Account'
        elif self.value in CryptoExchange.__members__:
            class_name = 'CryptoAccount'
            return getattr(importlib.import_module('pfund.accounts.account_crypto'), class_name)
        else:
            class_name = f'{self.capitalize()}Account'
        Account = getattr(importlib.import_module(f'pfund.accounts.account_{self.lower()}'), class_name)
        return Account