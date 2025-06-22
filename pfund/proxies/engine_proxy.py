from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.engines.base_engine_settings import BaseEngineSettings
    from pfund.datas.databoy import DataBoy
    from pfund.enums import TradingVenue
    from pfund.brokers.broker_base import BaseBroker


class EngineProxy:
    '''
    Proxy of BaseEngine when running in remote mode, since BaseEngine is not serializable by Ray.
    '''
    def __init__(self, databoy: DataBoy, settings: BaseEngineSettings):
        self._databoy = databoy
        self._settings = settings
        
    @property
    def settings(self) -> BaseEngineSettings:
        return self._settings
    
    def add_broker(self, trading_venue: TradingVenue) -> BaseBroker:
        pass

    # TODO: send back zmq ports in use to engine, and get the latest updated settings from engine
    # databoy.get_zmq_ports_in_use()
    