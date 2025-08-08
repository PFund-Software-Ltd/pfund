from pfeed.enums import DataCategory
from pfund.datas.data_time_based import TimeBasedData


class MarketData(TimeBasedData):
    @property
    def category(self) -> DataCategory:
        return DataCategory.MARKET_DATA
    
    # TODO: create MarketDataMetadata class (typed dict/dataclass/pydantic model)
    def to_dict(self) -> dict:
        return {
            'data_source': self.source.value,
            'data_origin': self.origin,
            'data_category': self.category.value,
            'product': str(self.product),
            'resolution': repr(self.resolution),
        }
    
    def __str__(self):
        return f'Data={self.product}|{self.resolution}'
