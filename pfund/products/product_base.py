from abc import ABC, abstractmethod


class BaseProduct(ABC):
    @staticmethod
    def create_product_name(*args, **kwargs):
        return '_'.join([str(arg) for arg in args])
    
    # TODO
    @staticmethod
    def parse_product_name(pdt):
        return pdt.split('_')
    
    def __init__(
        self, 
        bkr: str, 
        exch: str, 
        base_currency: str, 
        quote_currency: str, 
        product_type: str,
        *args,
        **kwargs,
    ):
        self.bkr = bkr.upper()
        self.exch = exch.upper()
        self.base_currency = self.bccy = base_currency.upper()
        self.quote_currency = self.qccy = quote_currency.upper()
        self.product_type = self.ptype = product_type.upper()
        self.currency_pair = self.pair = self.base_currency + '_' + self.quote_currency
        self.name = self.pdt = self.create_product_name(self.base_currency, self.quote_currency, self.product_type, *args, **kwargs)

    @abstractmethod
    def is_crypto(self) -> bool:
        pass

    def __str__(self):
        return f'Broker={self.bkr}|Exchange={self.exch}|Product={self.pdt}'

    def __repr__(self):
        return f'{self.bkr}-{self.exch}-{self.pdt}'
    
    def __eq__(self, other):
        if not isinstance(other, BaseProduct):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self.bkr == other.bkr
            and self.exch == other.exch
            and self.pdt == other.pdt
        )
        
    def __hash__(self):
        return hash((self.bkr, self.exch, self.pdt))