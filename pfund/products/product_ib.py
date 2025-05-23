from typing import Any

from decimal import Decimal

# FIXME: use pfund-ibapi
from pfund.externals.ibapi.contract import Contract
from pfund.enums import TradingVenue, Broker, TraditionalAssetType
from pfund.products.product_base import BaseProduct


# FIXME
# product types that will contribute to your total assets
PRODUCT_TYPES_AS_ASSETS = ['CRYPTO', 'STK', 'ETF', 'OPT', 'FX', 'BOND', 'MTF', 'CMDTY']  


class IBProduct(Contract, BaseProduct):
    trading_venue: TradingVenue = TradingVenue.IB
    broker: Broker = Broker.IB
    exchange: str=''

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        self.exchange = self.exchange.upper()

    @staticmethod
    def create_product_name(bccy, qccy, ptype, *args, **kwargs):
        from pfund.utils.utils import convert_to_uppercases
        bccy, qccy, ptype, *args = convert_to_uppercases(bccy, qccy, ptype, *args)
        name = TradFiProduct.create_product_name(bccy, qccy, ptype, *args, **kwargs)
        if ptype in ['FUT', 'OPT']:
            
            if 'lastTradeDateOrContractMonth' in kwargs:
                name += '_' + kwargs['lastTradeDateOrContractMonth']
            elif 'localSymbol' in kwargs:
                name += '_' + kwargs['localSymbol']
            else:
                raise Exception(f'IB product {name} is missing "lastTradeDateOrContractMonth" or "localSymbol"')
            
            if ptype == 'OPT':
                assert 'strike' in kwargs
                assert 'right' in kwargs  # option's right, "C" or "P"
                name += '_'.join([kwargs['right'], kwargs['strike']])
        return name

    # TODO
    @staticmethod
    def parse_product_name(pdt):
        return pdt.upper().split('_')
    
    def is_asset(self):
        return True if self.product_type in self._PRODUCT_TYPES_AS_ASSETS else False

    def __init__(
        self, 
        exch: str,
        base_currency: str,
        quote_currency: str,
        product_type: str,
        *args, 
        **kwargs
    ):
        TradFiProduct.__init__(self, 'IB', exch, base_currency, quote_currency, product_type, *args, **kwargs)
        Contract.__init__(self)  # inherits attributes from `Contract` (official class from IB)
        assert self.product_type in TradFiProductType.__members__, f'{self.product_type} is not supported, supported product types: {list(TradFiProductType.__members__)}'

        self.symbol = self.base_currency
        self.currency = self.quote_currency
        self.secType = self.product_type
        
        # set kwargs for IB contract
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # EXTEND
        self.exchange = self.exch
        if self.ptype == 'STK':
            if self.exch != 'SMART':  # if specified by user
                self.primaryExchange = self.exch.upper()
                self.exchange = 'SMART'
                
        # FIXME
        # self.name = self.pdt = self.create_product_name(exch, base_currency, quote_currency, product_type, *args, **kwargs)

    def is_crypto(self):
        return True if self.secType == 'CRYPTO' else False
