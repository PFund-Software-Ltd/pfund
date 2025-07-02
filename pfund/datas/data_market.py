from pfeed.enums import DataCategory
from pfund.datas.data_time_based import TimeBasedData


class MarketData(TimeBasedData):
    @property
    def category(self) -> DataCategory:
        return DataCategory.MARKET_DATA
    
    @property
    def zmq_channel(self) -> str:
        return f'{self.product.tv}:{repr(self.resolution)}:{self.product.name}:'
    
    @property
    def key(self) -> str:
        return f'{self.source}:{self.origin}:{self.product.name}:{repr(self.resolution)}'
    
    # TODO: create MarketDataMetadata class (typed dict/dataclass/pydantic model)
    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'data_source': self.source.value,
            'data_origin': self.origin,
            'data_category': self.category.value,
            'zmq_channel': self.zmq_channel,
            'product': str(self.product),
            'resolution': repr(self.resolution),
        }
    
    def __str__(self):
        return f'{self.product}|Data={self.resolution}'
