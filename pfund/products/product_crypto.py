from decimal import Decimal

from pfund.products.product_base import BaseProduct
from pfund.const.common import SUPPORTED_CRYPTO_PRODUCT_TYPES
from pfund.const.common import SUPPORTED_CRYPTO_MONTH_CODES
from pfund.const.common import CRYPTO_PRODUCT_TYPES_WITH_MATURITY


class CryptoProduct(BaseProduct):
    def __init__(
        self, 
        exch: str, 
        base_currency: str, 
        quote_currency: str, 
        product_type: str,
        *args,
        **kwargs
    ):
        super().__init__('CRYPTO', exch, base_currency, quote_currency, product_type, *args, **kwargs)
        assert self.product_type in SUPPORTED_CRYPTO_PRODUCT_TYPES
        # `args` will be joined by '_' to form a complete `name`, e.g. BTC_USDT_FUT_CM
        self.name = self.pdt = '_'.join(self.name.split('_') + list(args))
        # used by exchanges like bybit, okx, e.g., bybit has 4 categories ['linear', 'inverse', 'spot', 'option']
        self.category = ''  
        self.month_code = self._extract_month_code(args)
        
        self.taker_fee = self.tfee = None
        self.maker_fee = self.mfee = None
        self.tick_size = self.tsize = None
        self.lot_size = self.lsize = None
        self.multiplier = self.multi = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _extract_month_code(self, args):
        if self.product_type in CRYPTO_PRODUCT_TYPES_WITH_MATURITY and args:
            month_code = args[0]
            assert month_code in SUPPORTED_CRYPTO_MONTH_CODES, \
            f'{self.exch} {self.pdt} month code is invalid, valid options are: {SUPPORTED_CRYPTO_MONTH_CODES}'
            return month_code
        else:
            return None

    def is_crypto(self) -> bool:
        return True

    def is_spot(self) -> bool:
        return (self.ptype == 'SPOT')
    
    def is_option(self) -> bool:
        return (self.ptype == 'OPT')

    def is_inverse(self) -> bool:
        """Returns True if the product is an e.g. inverse perpetual contract (IPERP)"""
        return (self.ptype in ['IPERP', 'IFUT'])

    def load_configs(self, configs):
        # load product specs, including fees, multiplier etc.
        specs = configs.load_config_section('specs')
        ptype_specs = specs[self.ptype] if not self.month_code else specs[self.ptype][self.month_code]
        self.tfee = Decimal(str(configs.load_all_and_except_config(ptype_specs['tfee'], self.ptype, self.pdt)))
        self.mfee = Decimal(str(configs.load_all_and_except_config(ptype_specs['mfee'], self.ptype, self.pdt)))
        self.multi = Decimal(str(configs.load_all_and_except_config(ptype_specs['multi'], self.ptype, self.pdt)) if 'multi' in ptype_specs else 1)
        # load tick_sizes and lot_sizes
        tick_sizes = configs.read_config('_'.join(['tick_sizes', self.category]))
        lot_sizes = configs.read_config('_'.join(['lot_sizes', self.category]))
        self.tsize = Decimal(str(tick_sizes[0][self.pdt.upper()]))
        self.lsize = Decimal(str(lot_sizes[0][self.pdt.upper()]))

    def get_fee(self, fee_type, in_bps=False):
        if fee_type == 'taker':
            fee = self.tfee
        elif fee_type == 'maker':
            fee = self.mfee
        if not in_bps:
            fee /= 10000
        return fee
