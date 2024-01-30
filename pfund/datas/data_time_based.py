from pfund.datas.data_base import BaseData
from pfund.datas.resolution import Resolution
from pfund.products.product_base import BaseProduct


class TimeBasedData(BaseData):
    def __init__(self, product: BaseProduct, resolution: Resolution):
        super().__init__(product)
        self.base = 'time'
        self.latency = self.lat = None
        self.resolution = resolution
        self.period = resolution.period
        self.timeframe = resolution.timeframe
        self._resamplers = []  # data used to be resampled into another data
        self._resamplees = []  # opposite of resampler

    def __repr__(self):
        return f'{repr(self.product)}-{repr(self.resolution)}'

    def __str__(self):
        return f'{self.product}|Data={self.resolution}'

    def __eq__(self, other):
        if not isinstance(other, TimeBasedData):
            return NotImplemented
        return (self.product, self.resolution) == (other.product, other.resolution)
    
    def __hash__(self):
        return hash((self.product, self.resolution))
    
    def is_time_based(self):
        return True

    def is_resamplee(self):
        return bool(self._resamplers)

    def is_resampler(self):
        return bool(self._resamplees)

    def get_resamplees(self):
        return self._resamplees
    
    def get_resamplers(self):
        return self._resamplers

    def add_resamplee(self, data):
        if data not in self._resamplees:
            self._resamplees.append(data)

    def add_resampler(self, data):
        if data not in self._resamplers:
            self._resamplers.append(data)

    def remove_resamplee(self, data):
        if data in self._resamplees:
            self._resamplees.remove(data)
        
    def remove_resampler(self, data):
        if data in self._resamplers:
            self._resamplers.remove(data)