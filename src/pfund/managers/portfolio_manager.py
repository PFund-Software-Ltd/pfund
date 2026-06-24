from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias, TypedDict

if TYPE_CHECKING:
    from pfund.enums import TradingVenue
    from pfund.typing import AccountName, ProductName, Currency
    from pfund.entities import BaseBalance, BasePosition
    from pfund.entities.positions.position_base import PositionUpdate
    from pfund.entities.balances.balance_base import BalanceUpdate


import logging


class PortfolioManager:
    def __init__(self):
        self._logger = logging.getLogger("pfund.portfolio_manager")
        self._positions: dict[AccountName, dict[ProductName, BasePosition]] = {}
        self._balances: dict[AccountName, dict[Currency, BaseBalance]] = {}

    def update_positions(self, update: PositionUpdate):
        pass

    def update_balances(self, update: BalanceUpdate):
        pass

    # TODO: also reconcile with strategies
    # def reconcile_positions(self):
    #     def work():
    #         for exch in self._accounts:
    #             for acc in self._accounts[exch]:
    #                 self.get_positions(exch, acc, is_api_call=True)

    #     func = inspect.stack()[0][3]
    #     Thread(target=work, name=func + "_thread", daemon=True).start()

    # TODO: also reconcile with strategies
    # def reconcile_balances(self):
    #     def work():
    #         for exch in self._accounts:
    #             for acc in self._accounts[exch]:
    #                 self.get_balances(exch, acc, is_api_call=True)

    #     func = inspect.stack()[0][3]
    #     Thread(target=work, name=func + "_thread", daemon=True).start()
