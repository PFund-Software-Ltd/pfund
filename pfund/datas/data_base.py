class BaseData:
    def __init__(self, product):
        self.bkr = product.bkr
        self.exch = product.exch
        self.pdt = product.name
        self.product = product
    
    def is_crypto(self):
        return self.product.is_crypto()

    def is_time_based(self):
        return False

    def __eq__(self, other):
        if not isinstance(other, BaseData):
            return NotImplemented
        return self.product == other.product
    
    def __hash__(self):
        return hash(self.product)