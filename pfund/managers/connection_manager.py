from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.brokers.broker_trade import TradeBroker

import time
from collections import defaultdict
from multiprocessing import Process, Value
try:
    import psutil
except ImportError:
    pass

from pfund.enums import Event, RunMode

def _start_process(api, stop_flag: Value):
    try:
        api.start_zmqs()
        api.connect()
        if hasattr(api, 'get_servers'):
            servers = api.get_servers()
            public_server: str = servers[0]  # choose the first public server
        else:
            public_server = api.name
        zmqs = [zmq for zmq in api.get_zmqs()]
        while not stop_flag.value:
            for zmq in zmqs:
                if msg := zmq.recv():
                    channel, topic, info = msg
                    # only use the zmq of public ws to respond back, i.e. no duplicate pongs
                    if channel == 0 and zmq.name == public_server:
                        if topic == 0:
                            api.pong()
                    elif channel == 1:
                        # TODO, using ws to place_o/cancel_o/amend_o
                        if topic == 1:
                            api.place_o(*info)
                        elif topic == 2:
                            api.cancel_o(*info)
                        elif topic == 3:
                            api.amend_o(*info)
                else:
                    time.sleep(0.001)  # avoid busy-waiting
        else:
            api.disconnect(reason='stop process')
            api.stop_zmqs()
    except:
        api.logger.exception(f'api {api.name} _start_process exception:')


class ConnectionManager:
    _PROCESS_NO_PONG_TOLERANCE_IN_SECONDS = 30

    def __init__(self, broker: TradeBroker):
        self._broker = broker
        self._logger = broker._logger
        self._is_connected = defaultdict(bool)
        self._is_reconnecting = defaultdict(bool)
        self._pids = defaultdict(lambda: None)
        self._apis = {}
        self._api_stop_flags = defaultdict(lambda: Value('b', False))
        self._api_procs = {}
        self._last_pong_ts = defaultdict(lambda: time.time())

    def _adjust_input_trading_venues(self, trading_venues: str|list[str]|None) -> list:
        if type(trading_venues) is str:
            trading_venues = [trading_venues]
        return trading_venues or list(self._apis)

    def get_trading_venues(self):
        return list(self._apis)

    def add_api(self, api):
        self._apis[api.name] = api
        self._logger.debug(f'added {api.name} api')

    def _set_pid(self, trading_venue: str, pid: int):
        prev_pid = self._pids[trading_venue]
        self._pids[trading_venue] = pid
        self._logger.debug(f'set {trading_venue} api process pid from {prev_pid} to {pid}')

    def _on_pong(self, trading_venue: str):
        """Handles pongs from api processes"""
        self._last_pong_ts[trading_venue] = time.time()
        self._logger.debug(f'{trading_venue} ponged')

    def is_process_healthy(self, trading_venue: str):
        if time.time() - self._last_pong_ts[trading_venue] >= self._PROCESS_NO_PONG_TOLERANCE_IN_SECONDS:
            self._logger.error(f'process {trading_venue} is not responding')
            return False
        else:
            return True

    # connected = ws is working properly (connected + authenticated + subscribed ...)
    def is_connected(self, trading_venue: str):
        return self._is_connected[trading_venue]

    def _on_connected(self, trading_venue: str):
        if not self._is_connected[trading_venue]:
            self._is_connected[trading_venue] = True
            self._logger.debug(f'{trading_venue} is connected')
        else:
            self._logger.warning(f'{trading_venue} is already connected')
    
    def _on_disconnected(self, trading_venue: str):
        if self._is_connected[trading_venue]:
            self._is_connected[trading_venue] = False
            self._logger.debug(f'{trading_venue} is disconnected')
        else:
            self._logger.warning(f'{trading_venue} is already disconnected')

    def _terminate_process(self, trading_venue: str):
        pid = self._pids[trading_venue]
        if pid is not None and psutil.pid_exists(pid):
            psutil.Process(pid).kill()
            self._logger.warning(f'force to terminate {trading_venue} process ({pid=})')
            self._set_pid(trading_venue, None)

    def connect(self, trading_venues: str|list[str]|None=None):
        trading_venues = self._adjust_input_trading_venues(trading_venues)
        for trading_venue in trading_venues:
            self._logger.debug(f'{trading_venue} is connecting')
            api = self._apis[trading_venue]
            stop_flag = self._api_stop_flags[trading_venue]
            stop_flag.value = False
            self._api_procs[trading_venue] = Process(target=_start_process, args=(api, stop_flag), name=f'{trading_venue}_process', daemon=True)
            self._api_procs[trading_venue].start()

    def disconnect(self, trading_venues: str|list[str]|None=None):
        trading_venues = self._adjust_input_trading_venues(trading_venues)
        for trading_venue in trading_venues:
            self._logger.debug(f'{trading_venue} is disconnecting')
            stop_flag = self._api_stop_flags[trading_venue]
            stop_flag.value = True
            # need to wait for the process to finish 
            # in case no pid has been returned (i.e. cannot terminate the process by pid)
            while self._api_procs[trading_venue].is_alive():
                self._logger.debug(f'waiting for api process {trading_venue} to finish')
                self._terminate_process(trading_venue)
                time.sleep(1)
            else:
                self._logger.debug(f'api process {trading_venue} is finished')
                del self._api_procs[trading_venue]
                self._on_disconnected(trading_venue)

    def reconnect(self, trading_venues: str|list[str]|None=None, reason: str=''):
        trading_venues = self._adjust_input_trading_venues(trading_venues)
        for trading_venue in trading_venues:
            if not self._is_reconnecting[trading_venue]:
                self._logger.debug(f'{trading_venue} is reconnecting ({reason=})')
                self._is_reconnecting[trading_venue] = True
                self.disconnect(trading_venue)
                self.connect(trading_venue)
                self._is_reconnecting[trading_venue] = False
            else:
                self._logger.warning(f'{trading_venue} is already reconnecting, do not reconnect again ({reason=})')

    def handle_msgs(self, topic, info):
        if topic == 0:  # pong
            bkr, exch, pong = info
            trading_venue = exch if bkr == 'CRYPTO' else bkr
            self._on_pong(trading_venue)
        elif topic == 1:
            bkr, exch, pid = info
            trading_venue = exch if bkr == 'CRYPTO' else bkr
            self._set_pid(trading_venue, pid)
        elif topic == 2:
            bkr, exch, connected = info
            trading_venue = exch if bkr == 'CRYPTO' else bkr
            self._on_connected(trading_venue)
        elif topic == 3:
            bkr, exch, disconnected = info
            trading_venue = exch if bkr == 'CRYPTO' else bkr
            self._on_disconnected(trading_venue)
