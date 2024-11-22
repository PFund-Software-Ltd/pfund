from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.accounts.account_base import BaseAccount
    from pfund.positions.position_base import BasePosition
    from pfund.balances.balance_base import BaseBalance

from collections import defaultdict

    
class BasePortfolio:
    def __init__(self):
        # EXTEND: if Currency class is ready, then find a way to group currencies by e.g. type (crypto, fiat), region, safety, ...
        # then create e.g. CryptoCurrencyMixin, FiatCurrencyMixin
        self._currencies = defaultdict(dict)  # {account: {ccy: balance}}
    
    @classmethod
    def from_positions_and_balances(
        cls,
        positions: list[BasePosition] | None=None,
        balances: list[BaseBalance] | None=None,
    ) -> BasePortfolio:
        portfolio = cls()
        positions = positions or []
        balances = balances or []
        for position in positions:
            portfolio.add_position(position)
        for balance in balances:
            portfolio.add_balance(balance)
        return portfolio
    
    def get_positions(
        self, 
        ptype: str, 
        account: BaseAccount | None=None, 
        pdt: str=''
    ) -> defaultdict[str, dict[str, BasePosition]] | dict[str, BasePosition] | BasePosition | None:
        assets = self._get_assets(ptype)
        pdt = pdt.upper()
        if not account and not pdt:
            return assets
        elif account and not pdt:
            return assets.get(account, None)
        elif account and pdt:
            return assets[account].get(pdt, None)
        else: # not account and pdt
            return {account: position for account, pdt_to_position in assets.items() for position in pdt_to_position.values() if position.pdt == pdt}
    
    def get_balances(
        self, 
        account: BaseAccount | None=None, 
        ccy: str=''
    ) -> defaultdict[str, dict[str, BaseBalance]] | dict[str, BaseBalance] | BaseBalance | None:
        balances = self._currencies
        ccy = ccy.upper()
        if not account and not ccy:
            return balances
        elif account and not ccy:
            return balances.get(account, None)
        elif account and ccy:
            return balances[account].get(ccy, None)
        else: # not account and ccy
            return {account: balance for account, ccy_to_balance in balances.items() for balance in ccy_to_balance.values() if balance.ccy == ccy}
        
    def add_position(self, position: BasePosition):
        assets: dict = self._get_assets(position.ptype)
        assets[position.account][position.pdt] = position

    def add_balance(self, balance: BaseBalance):
        self._currencies[balance.account][balance.ccy] = balance
    
    def update_position(self, position: BasePosition):
        assets: dict = self._get_assets(position.ptype)
        if self.has_position(position):
            assets[position.account][position.pdt] = position
        else:
            raise ValueError(f'{position} not in {assets}')
    
    def update_balance(self, balance: BaseBalance):
        if self.has_balance(balance):
            self._currencies[balance.account][balance.ccy] = balance
        else:
            raise ValueError(f'{balance} not in {self._currencies}')
        
    def has_position(self, position: BasePosition) -> bool:
        assets: dict = self._get_assets(position.ptype)
        return position.account in assets and position.pdt in assets[position.account]
    
    def has_balance(self, balance: BaseBalance) -> bool:
        return balance.account in self._currencies and balance.ccy in self._currencies[balance.account]
    
    def remove_position(self, position: BasePosition):
        assets: dict = self._get_assets(position.ptype)
        if self.has_position(position):
            del assets[position.account][position.pdt]
        else:
            raise ValueError(f'{position} not in {assets}')
    
    def remove_balance(self, balance: BaseBalance):
        if self.has_balance(balance):
            del self._currencies[balance.account][balance.ccy]
        else:
            raise ValueError(f'{balance} not in {self._currencies}')
    
     # TODO: add more functionalities, e.g.
    # - get_total_exposure(unit='USD') (rmb to include balances)
    # - get_positions_by_exposure()
    # - get_exposures_by_asset_class()
    # - ...
    