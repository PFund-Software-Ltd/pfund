from collections import defaultdict


class DefiAssetsMixin:
    def setup_assets(self):
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
        
        self._all_assets = {
            # ptype: assets
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
