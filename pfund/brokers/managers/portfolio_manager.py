from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from decimal import Decimal
    from pfund.typing import AccountName, Currency, ProductName
    from pfund.brokers.broker_base import BaseBroker
    from pfund.entities.balances.balance_base import BaseBalance
    from pfund.entities.positions.position_base import BasePosition
    from pfund.enums import TradingVenue

import logging


class PortfolioManager:
    _balances: dict[TradingVenue, dict[AccountName, dict[Currency, BaseBalance]]]
    _positions: dict[TradingVenue, dict[AccountName, dict[ProductName, BasePosition]]]

    def __init__(self, broker: BaseBroker):
        self._broker: BaseBroker = broker
        self._logger: logging.Logger = logging.getLogger("pfund")
        self._balances = {}
        self._positions = {}

    def get_balances(
        self, tv: TradingVenue, acc: AccountName = "", ccy: Currency = ""
    ) -> (
        dict[AccountName, dict[Currency, BaseBalance]]
        | list[BaseBalance]
        | dict[Currency, BaseBalance]
        | BaseBalance
        | None
    ):
        try:
            if not acc:
                balances_per_venue: dict[AccountName, dict[Currency, BaseBalance]] = self._balances[tv]
                if not ccy:
                    return balances_per_venue
                else:
                    balances_per_ccy: list[BaseBalance] = [
                        balance
                        for balances_per_account in balances_per_venue.values()
                        for _ccy, balance in balances_per_account.items()
                        if _ccy == ccy
                    ]
                    return balances_per_ccy
            else:
                balances_per_account: dict[Currency, BaseBalance] = self._balances[tv][acc]
                if not ccy:
                    return balances_per_account
                else:
                    return balances_per_account[ccy]
        except KeyError:
            return None

    def get_positions(
        self, tv: TradingVenue, acc: AccountName="", pdt: ProductName=""
    ) -> (
        dict[AccountName, dict[ProductName, BasePosition]]
        | list[BasePosition]
        | dict[ProductName, BasePosition]
        | BasePosition
        | None
    ):
        try:
            if not acc:
                positions_per_venue: dict[AccountName, dict[ProductName, BasePosition]] = self._positions[tv]
                if not pdt:
                    return positions_per_venue
                else:
                    positions_per_pdt: list[BasePosition] = [
                        position
                        for positions_per_account in positions_per_venue.values()
                        for _pdt, position in positions_per_account.items()
                        if _pdt == pdt
                    ]
                    return positions_per_pdt
            else:
                positions_per_account: dict[ProductName, BasePosition] = self._positions[tv][acc]
                if not pdt:
                    return positions_per_account
                else:
                    return positions_per_account[pdt]
        except KeyError:
            return None

    def add_balance(self, tv: TradingVenue, acc: AccountName, balance: BaseBalance):
        if tv not in self._balances:
            self._balances[tv] = {}
        if acc not in self._balances[tv]:
            self._balances[tv][acc] = {}
        ccy = balance.ccy
        if ccy in self._balances[tv][acc]:
            raise ValueError(f'{tv} balance {ccy} for account "{acc}" already exists')
        self._balances[tv][acc][ccy] = balance

    def add_position(self, tv: TradingVenue, acc: AccountName, position: BasePosition):
        if tv not in self._positions:
            self._positions[tv] = {}
        if acc not in self._positions[tv]:
            self._positions[tv][acc] = {}
        pdt = position.pdt
        if pdt in self._positions[tv][acc]:
            raise ValueError(f'{tv} position {pdt} for account "{acc}" already exists')
        self._positions[tv][acc][pdt] = position

    def update_balances(self, tv: TradingVenue, acc: AccountName, balances_snapshot: dict[str, Any]):
        ts: float = balances_snapshot["ts"]
        data: dict[Currency, dict[str, Decimal | float]] = balances_snapshot["data"]
        for ccy, update in data.items():
            balance: BaseBalance = cast(BaseBalance, self.get_balances(tv=tv, acc=acc, ccy=ccy))
            balance.on_update(update, ts=ts)

    def update_positions(self, tv: TradingVenue, acc: AccountName, positions_snapshot: dict[str, Any]):
        ts: float = positions_snapshot["ts"]
        data: dict[ProductName, dict[str, Decimal | float]] = positions_snapshot["data"]
        for pdt, update in data.items():
            position: BasePosition = cast(BasePosition, self.get_positions(tv=tv, acc=acc, pdt=pdt))
            position.on_update(update, ts=ts)
