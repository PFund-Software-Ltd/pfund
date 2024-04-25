from pathlib import Path

from pfund.exchanges.exchange_base import BaseExchange
        
        
class Exchange(BaseExchange):
    SUPPORTED_CATEGORIES = ['linear', 'inverse', 'spot', 'option']
    PTYPE_TO_CATEGORY = {
        'PERP': 'linear',
        'FUT': 'linear',
        'IPERP': 'inverse',
        'IFUT': 'inverse',
        'SPOT': 'spot',
        'OPT': 'option',
    }
    def __new__(cls, env: str, ptype: str):
        from pfund.exchanges.binance.linear.exchange import ExchangeLinear
        
        ptype = ptype.upper()
        category = cls.PTYPE_TO_CATEGORY[ptype]
        
        if category == 'linear':
            instance = super().__new__(ExchangeLinear)
            instance.category = category
            return instance
        # EXTEND: Add other categories
        else:
            raise ValueError(f"Invalid {category=}")
    
    def __init__(self, env: str, ptype: str):
        exch = Path(__file__).parent.name
        super().__init__(env, exch)
        
    # FIXME: temporarily override the method, remove it later
    def _setup_configs(self):
        pass
    
