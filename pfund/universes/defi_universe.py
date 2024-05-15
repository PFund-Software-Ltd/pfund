from pfund.universes.base_universe import BaseUniverse


class DefiUniverse(BaseUniverse):
    def __init__(self):
        super().__init__()
        
        # FIXME: drafted by ChatGPT
        self.liquidity_pools = {} 
        self.lending_platforms = {}
        self.yield_farms = {}
        self.governance_tokens = {}
        self.synthetic_assets = {}
        self.stablecoins = {}
        self.derivatives = {}
        self.insurance = {}
        self.indexes = {}
        self.nfts = {}
        
        self._assets = {
            # ptype: asset_class
            'LIQUIDITY_POOL': self.liquidity_pools,
            'LENDING_PLATFORM': self.lending_platforms,
            'YIELD_FARM': self.yield_farms,
            'GOVERNANCE_TOKEN': self.governance_tokens,
            'SYNTHETIC_ASSET': self.synthetic_assets,
            'STABLECOIN': self.stablecoins,
            'DERIVATIVE': self.derivatives,
            'INSURANCE': self.insurance,
            'INDEX': self.indexes,
            'NFT': self.nfts,
        }