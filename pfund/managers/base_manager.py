from collections import defaultdict

from pfund.datas.data_base import BaseData
from pfund.strategies.strategy_base import BaseStrategy
from pfund.plogging import create_dynamic_logger


class BaseManager:
    def __init__(self, name, broker):
        self.name = name
        self._broker = broker
        self._zmq = self._broker.get_zmq()
        self.logger = create_dynamic_logger(name, 'manager')
        self._listeners = defaultdict(list)

    def _add_listener(self, listener: BaseStrategy | BaseData, listener_key: BaseData | str):
        if listener not in self._listeners[listener_key]:
            self._listeners[listener_key].append(listener)

    def _remove_listener(self, listener: BaseStrategy | BaseData, listener_key: BaseData | str):
        if listener in self._listeners[listener_key]:
            self._listeners[listener_key].remove(listener)

    def _is_crypto_broker(self):
        return self._broker.name == 'CRYPTO'
    
    def run_regular_tasks(self):
        pass