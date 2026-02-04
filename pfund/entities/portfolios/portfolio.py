from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.positions.position_base import BasePosition
    from pfund.balances.balance_base import BaseBalance

from collections import defaultdict

from pfund.portfolios import BasePortfolio, CeFiPortfolio, DeFiPortfolio, TradFiPortfolio
from pfund.mixins.assets.all_assets_mixin import AllAssetsMixin


class Portfolio(AllAssetsMixin, BasePortfolio):
    '''A (unified) portfolio that combines multiple sub-portfolios from different brokers.'''
    def __init__(self):
        BasePortfolio.__init__(self)
        self._sub_portfolios = {}  # {bkr: portfolio}
        self.setup_assets()
        
    def initialize(self, positions: list[BasePosition], balances: list[BaseBalance]):
        for bkr in {pb.bkr for pb in positions + balances}:
            positions_per_bkr = [position for position in positions if position.bkr == bkr]
            balances_per_bkr = [balance for balance in balances if balance.bkr == bkr]
            portfolio: BasePortfolio = self._add_sub_portfolio(bkr, positions_per_bkr, balances_per_bkr)
            # e.g. allows using 'portfolio.crypto' to access CeFiPortfolio
            setattr(self, bkr.lower(), portfolio)
        
        # TODO: use global() to dynamically create attributes?
        for attr in (
            'stocks', 
            'futures', 
            'options', 
            'cashes', 
            'cryptos', 
            'bonds', 
            'funds', 
            'cmdties', 
            'perps', 
            'iperps', 
            'ifutures',
            '_currencies',  # REVIEW
        ):
            # combine assets from sub-portfolios, e.g. self.futures = futures in crypto portfolio + futures in tradfi portfolio
            setattr(self, attr, self.combine_dicts(*(getattr(pfo, attr) for pfo in self._sub_portfolios.values() if hasattr(pfo, attr))))
            if attr == 'cryptos':
                self.spots = self.cryptos
        
    @staticmethod
    # Function to combine nested dictionaries without copying
    def combine_dicts(*dicts: defaultdict[str, dict]) -> defaultdict[str, dict]:
        combined = defaultdict(dict)
        for d in dicts:
            for key, sub_dict in d.items():
                if key not in combined:
                    combined[key] = sub_dict
                else:
                    combined[key].update(sub_dict)
        return combined

    def _add_sub_portfolio(self, bkr: str, positions: list[BasePosition], balances: list[BaseBalance]) -> BasePortfolio:
        if bkr not in self._sub_portfolios:
            if bkr == 'CRYPTO':
                portfolio = CeFiPortfolio.from_positions_and_balances(positions=positions, balances=balances)
            elif bkr == 'DEFI':
                portfolio = DeFiPortfolio.from_positions_and_balances(positions=positions, balances=balances)
            else:
                portfolio = TradFiPortfolio.from_positions_and_balances(positions=positions, balances=balances)
            self._sub_portfolios[bkr] = portfolio
        return self._sub_portfolios[bkr]