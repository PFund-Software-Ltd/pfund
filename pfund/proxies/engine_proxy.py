from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.datas.databoy import DataBoy
    from pfund.enums import tTradingVenue
    from pfund.brokers.broker_base import BaseBroker


class EngineProxy:
    '''
    Proxy of BaseEngine when running in remote mode, since BaseEngine is not serializable by Ray.
    '''
    def __init__(self, databoy: DataBoy):
        self._databoy = databoy
    
    def add_broker(self, trading_venue: tTradingVenue) -> BaseBroker:
        pass