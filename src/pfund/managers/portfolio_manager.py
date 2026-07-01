from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pfund.typing import AccountName, ProductName, Currency
    from pfund.entities.balances.balance_base import BaseBalance, BalanceUpdate
    from pfund.entities.positions.position_base import BasePosition


import logging


class PortfolioManager:
    def __init__(self):
        self._logger = logging.getLogger("pfund.portfolio_manager")
        self._positions: dict[AccountName, dict[ProductName, BasePosition]] = {}
        self._balances: dict[AccountName, dict[Currency, BaseBalance]] = {}

    @property
    def balances(self) -> dict[AccountName, dict[Currency, BaseBalance]]:
        return self._balances

    @property
    def positions(self) -> dict[AccountName, dict[ProductName, BasePosition]]:
        return self._positions

    def on_balance_update(self, update: BalanceUpdate[Any]):
        if update.source == "get_balances":
            self.reconcile_balances()
        else:
            self.update_balances(update.account, update.snapshots)

    def on_position_update(self, update: PositionUpdate[Any]):
        if update.source == "get_positions":
            self.reconcile_positions()
        else:
            self.update_positions(update.account, update.snapshots)

    # TODO: check if the update ts is newer than the existing snapshot
    def update_balances(
        self, account: AccountName, snapshots: dict[Currency, BaseBalance.Snapshot]
    ):
        for currency, snapshot in snapshots.items():
            self._balances[account][currency].on_update(snapshot)

    # TODO: check if the update ts is newer than the existing snapshot
    def update_positions(
        self, account: AccountName, snapshots: dict[ProductName, BasePosition.Snapshot]
    ):
        if account not in self._positions:
            self._positions[account] = {}
        positions = self._positions[account]
        for product, snapshot in snapshots.items():
            if product not in positions:
                positions[product] = BasePosition()
            positions[product].on_update(snapshot)

    # TODO
    def reconcile_positions(self):
        pass

    # TODO
    def reconcile_balances(self):
        pass
