from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.positions.position_base import BasePosition
    from pfund.balances.balance_base import BaseBalance

from collections import defaultdict

from rich.console import Console

from pfund.portfolios import BasePortfolio, CryptoPortfolio, DefiPortfolio, TradfiPortfolio
from pfund.const.common import SUPPORTED_TRADFI_PRODUCT_TYPES, SUPPORTED_CRYPTO_PRODUCT_TYPES
from pfund.mixins.assets import TradfiAssetsMixin, CryptoAssetsMixin, DefiAssetsMixin


class Portfolio(TradfiAssetsMixin, CryptoAssetsMixin, DefiAssetsMixin, BasePortfolio):
    '''A (unified) portfolio that combines multiple sub-portfolios from different brokers.'''
    def __init__(self):
        BasePortfolio.__init__(self)
        self._sub_portfolios = {}  # {bkr: portfolio}
        all_assets = {}
        TradfiAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        CryptoAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        DefiAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        self._all_assets = all_assets
        
    def initialize(self, positions: list[BasePosition], balances: list[BaseBalance]):
        for bkr in {position.bkr for position in positions}:
            positions_per_bkr = [position for position in positions if position.bkr == bkr]
            portfolio: BasePortfolio = self._add_sub_portfolio(bkr, positions_per_bkr)
            setattr(self, bkr.lower(), portfolio)

    def _get_assets(self, ptype: str) -> defaultdict[str, dict[str, BasePosition]]:
        ptype = ptype.upper()
        # TODO: add SUPPORTED_DEFI_PRODUCT_TYPES
        if ptype not in SUPPORTED_TRADFI_PRODUCT_TYPES + SUPPORTED_CRYPTO_PRODUCT_TYPES:
            raise KeyError(f'Invalid {ptype=}, supported choices: {SUPPORTED_TRADFI_PRODUCT_TYPES+SUPPORTED_CRYPTO_PRODUCT_TYPES}')
        else:
            return self._all_assets[ptype]
    
    def _add_sub_portfolio(self, bkr: str, positions: list[BasePosition]) -> BasePortfolio:
        if bkr not in self._sub_portfolios:
            if bkr == 'CRYPTO':
                portfolio = CryptoPortfolio.from_positions_and_balances(positions)
            elif bkr == 'DEFI':
                portfolio = DefiPortfolio.from_positions_and_balances()
            else:
                portfolio = TradfiPortfolio.from_positions_and_balances()
            self._sub_portfolios[bkr] = portfolio
        return self._sub_portfolios[bkr]
    
    # TODO: add more functionalities, e.g.
    # - get_total_exposure(unit='USD')
    # - get_positions_by_exposure()
    # - get_exposures_by_asset_class()
    # - ...
    
    
    