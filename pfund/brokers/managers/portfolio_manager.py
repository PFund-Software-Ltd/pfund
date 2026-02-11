from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar, cast

if TYPE_CHECKING:
    from pfund.typing import AccountName, Currency, ProductName
    from pfund.entities.balances.balance_update import BalanceUpdate
    from pfund.enums import TradingVenue

from pfund.entities.balances.balance_base import BaseBalance
from pfund.entities.positions.position_base import BasePosition

import logging

Balance = TypeVar('Balance', bound=BaseBalance)
Position = TypeVar('Position', bound=BasePosition)


class PortfolioManager(Generic[Balance, Position]):
    _balances: dict[TradingVenue, dict[AccountName, dict[Currency, Balance]]]
    _positions: dict[TradingVenue, dict[AccountName, dict[ProductName, Position]]]

    def __init__(self):
        self._logger: logging.Logger = logging.getLogger("pfund")
        self._balances = {}
        self._positions = {}

    def get_balances(
        self, tv: TradingVenue, acc: AccountName = "", ccy: Currency = ""
    ) -> (
        dict[AccountName, dict[Currency, Balance]]
        | list[Balance]
        | dict[Currency, Balance]
        | Balance
        | None
    ):
        try:
            if not acc:
                balances_per_venue = self._balances[tv]
                if not ccy:
                    return balances_per_venue
                else:
                    balances_per_ccy: list[Balance] = [
                        balance
                        for balances_per_account in balances_per_venue.values()
                        for _ccy, balance in balances_per_account.items()
                        if _ccy == ccy
                    ]
                    return balances_per_ccy
            else:
                balances_per_account = self._balances[tv][acc]
                if not ccy:
                    return balances_per_account
                else:
                    return balances_per_account[ccy]
        except KeyError:
            return None

    def get_positions(
        self, tv: TradingVenue, acc: AccountName="", pdt: ProductName=""
    ) -> (
        dict[AccountName, dict[ProductName, Position]]
        | list[Position]
        | dict[ProductName, Position]
        | Position
        | None
    ):
        try:
            if not acc:
                positions_per_venue = self._positions[tv]
                if not pdt:
                    return positions_per_venue
                else:
                    positions_per_pdt: list[Position] = [
                        position
                        for positions_per_account in positions_per_venue.values()
                        for _pdt, position in positions_per_account.items()
                        if _pdt == pdt
                    ]
                    return positions_per_pdt
            else:
                positions_per_account = self._positions[tv][acc]
                if not pdt:
                    return positions_per_account
                else:
                    return positions_per_account[pdt]
        except KeyError:
            return None

    def add_balance(self, tv: TradingVenue, acc: AccountName, balance: Balance):
        if tv not in self._balances:
            self._balances[tv] = {}
        if acc not in self._balances[tv]:
            self._balances[tv][acc] = {}
        ccy = balance.ccy
        if ccy in self._balances[tv][acc]:
            raise ValueError(f'{tv} balance {ccy} for account "{acc}" already exists')
        self._balances[tv][acc][ccy] = balance

    def add_position(self, tv: TradingVenue, acc: AccountName, position: Position):
        if tv not in self._positions:
            self._positions[tv] = {}
        if acc not in self._positions[tv]:
            self._positions[tv][acc] = {}
        pdt = position.pdt
        if pdt in self._positions[tv][acc]:
            raise ValueError(f'{tv} position {pdt} for account "{acc}" already exists')
        self._positions[tv][acc][pdt] = position

    def update_balances(self, tv: TradingVenue, acc: AccountName, update: BalanceUpdate):
        for ccy, data in update.data.items():
            balance: Balance = cast(Balance, self.get_balances(tv=tv, acc=acc, ccy=ccy))
            balance.on_update(data, ts=update.ts)
            # TODO: detect balance changes and log them, do NOT log the whole snapshot

    def update_positions(self, tv: TradingVenue, acc: AccountName, update: PositionUpdate):
        for pdt, data in update.data.items():
            position: Position = cast(Position, self.get_positions(tv=tv, acc=acc, pdt=pdt))
            position.on_update(data, ts=update.ts)
            # TODO: detect position changes and log them, do NOT log the whole snapshot
