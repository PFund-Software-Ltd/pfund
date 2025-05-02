from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler
    from pfund.brokers.broker_trade import BaseBroker


class BaseManager:
    def __init__(self, name: str, broker: BaseBroker):
        from pfund._logging import create_dynamic_logger
        self.name = name.lower()
        self._broker = broker
        self._zmq = broker.get_zmq()
        self.logger = create_dynamic_logger(name, 'manager')
    
    def schedule_jobs(self, scheduler: BackgroundScheduler):
        pass