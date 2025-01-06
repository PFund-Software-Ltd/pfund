"""This is an engine used to trade against multiple brokers and cryptocurrency exchanges.

This engine is designed for algorithmic trading and contains all the major 
components at the highest level such as:
    brokers (e.g. Interactive Brokers, Crypto, ...),
        where broker `Crypto` is a fake broker name that includes the actual
        crypto exchanges (e.g. Binance, Bybit, ...)
    strategies (your trading strategies)
In order to communicate with other processes, it uses ZeroMQ as the core 
message queue.
"""
from __future__ import annotations

import time
from threading import Thread

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfeed.typing.literals import tDATA_TOOL

import schedule

from pfund.engines.base_engine import BaseEngine
from pfund.brokers.broker_base import BaseBroker
from pfund.utils.utils import flatten_dict, is_port_in_use


class TradeEngine(BaseEngine):
    zmq_ports = {}

    def __new__(
        cls, 
        *, 
        env: Literal['SANDBOX', 'PAPER', 'LIVE']='PAPER', 
        data_tool: tDATA_TOOL='pandas', 
        df_min_rows: int=1_000,
        df_max_rows: int=3_000,
        zmq_port=5557, 
        **settings
    ):
        if not hasattr(cls, 'zmq_port'):
            assert isinstance(zmq_port, int), f'{zmq_port=} must be an integer'
            cls._zmq_port = zmq_port
        instance = super().__new__(
            cls,
            env,
            data_tool=data_tool,
            **settings
        )
        if not hasattr(cls, 'df_min_rows'):
            cls.DataTool.set_min_rows(df_min_rows)
        if not hasattr(cls, 'df_max_rows'):
            cls.DataTool.set_max_rows(df_max_rows)
        return instance

    def __init__(
        self,
        *,
        env: Literal['SANDBOX', 'PAPER', 'LIVE']='PAPER',
        data_tool: tDATA_TOOL='pandas',
        df_min_rows: int=1_000,
        df_max_rows: int=3_000,
        zmq_port=5557,
        **settings
    ):
        from pfund.zeromq import ZeroMQ

        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            self._is_running = True
            self._zmq = ZeroMQ('engine')
            self._background_thread = None
            super().__init__(
                env,
                data_tool=data_tool,
                **settings
            )

    @classmethod
    def assign_cpus(cls, name) -> list:
        if 'cpu_affinity' in cls.settings and name in cls.settings['cpu_affinity']:
            assigned_cpus = cls.settings['cpu_affinity'][name]
        else:
            assigned_cpus = []
        if not isinstance(assigned_cpus, list):
            assigned_cpus = [assigned_cpus]
        return assigned_cpus
    
    def _assign_zmq_ports(self) -> dict:
        _assigned_ports = []
        def _is_port_available(_port):
            _is_port_assigned = (_port in _assigned_ports)
            if is_port_in_use(_port) or _is_port_assigned:
                return False
            else:
                _assigned_ports.append(_port)
                return True
        def _get_port(start_port=None):
            _port = start_port or self._zmq_port
            if _is_port_available(_port):
                return _port
            else:
                return _get_port(start_port=_port+1)
        self.zmq_ports['engine'] = _get_port()
        for broker in self.brokers.values():
            if broker.name == 'CRYPTO':
                for exchange in broker.exchanges.values():
                    self.zmq_ports[exchange.name] = {'rest_api': _get_port()}
                    if not exchange.use_separate_private_ws_url():
                        self.zmq_ports[exchange.name]['ws_api'] = _get_port()
                    else:
                        self.zmq_ports[exchange.name]['ws_api'] = {'public': {}, 'private': {}}
                        ws_servers = exchange.get_ws_servers()
                        for ws_server in ws_servers:
                            self.zmq_ports[exchange.name]['ws_api']['public'][ws_server] = _get_port()
                        for acc in exchange.accounts.keys():
                            self.zmq_ports[exchange.name]['ws_api']['private'][acc] = _get_port()
            else:
                self.zmq_ports[broker.name] = _get_port()
        for strategy in self.strategy_manager.strategies.values():
            if strategy.is_parallel():
                self.zmq_ports[strategy.name] = _get_port()
        self.logger.debug(f'{self.zmq_ports=}')

    def _schedule_background_tasks(self):
        schedule.every(self._PROCESS_NO_PONG_TOLERANCE_IN_SECONDS//3).seconds.do(self._ping_processes)
        schedule.every(self._PROCESS_NO_PONG_TOLERANCE_IN_SECONDS).seconds.do(self._check_processes)
        schedule.every(10).seconds.do(self.run_regular_tasks)

    def _run_background_tasks(self):
        while self._is_running:
            schedule.run_pending()
            time.sleep(3)
    
    def _ping_processes(self):
        self._zmq.send(0, 0, ('engine', 'ping',))

    def _check_processes(self):
        for broker in self.brokers.values():
            connection_manager = broker.cm
            trading_venues = connection_manager.get_trading_venues()
            if reconnect_trading_venues := [trading_venue for trading_venue in trading_venues if not connection_manager.is_process_healthy(trading_venue)]:
                connection_manager.reconnect(reconnect_trading_venues, reason='process not responding')
        if restart_strats := [strat for strat, strategy in self.strategy_manager.strategies.items() if strategy.is_parallel() and not self.strategy_manager.is_process_healthy(strat)]:
            self.strategy_manager.restart(restart_strats, reason='process not responding')

    def run_regular_tasks(self):
        for broker in self.brokers.values():
            broker.run_regular_tasks()
    
    def add_broker(self, bkr: str) -> BaseBroker:
        bkr = bkr.upper()
        if bkr in self.brokers:
            return self.get_broker(bkr)
        Broker = self.get_Broker(bkr)
        broker = Broker(self.env)
        self.brokers[bkr] = broker
        self.logger.debug(f'added {bkr=}')
        return broker
    
    def run(self):
        self._assign_zmq_ports()
        self._zmq.start(
            logger=self.logger,
            send_port=self.zmq_ports['engine'],
            recv_ports=[port for k, port in flatten_dict(self.zmq_ports).items() if k != 'engine']
        )

        for broker in self.brokers.values():
            broker.start(zmq=self._zmq)

        self.strategy_manager.start()
        
        self._schedule_background_tasks()
        schedule.run_all()  # run all tasks at start
        self._background_thread = Thread(target=self._run_background_tasks, daemon=True)
        self._background_thread.start()
        self.logger.debug('background thread started')

        # TEMP, zeromq examples
        # self._zmq.send(
        #     channel=999,
        #     topic=888,
        #     info="to private ws",
        #     receiver=(acc:='test'),
        # )
        # self._zmq.send(
        #     channel=123,
        #     topic=456,
        #     info="to strategy",
        #     receiver=(strat:='test_strategy'),
        # )

        while self._is_running:
            if msg := self._zmq.recv():
                channel, topic, info = msg
                # TODO
                if channel == 0:  # from strategy processes to strategy manager
                    self.strategy_manager.handle_msgs(topic, info)
                else:
                    bkr = info[0]
                    broker = self.brokers[bkr]
                    broker.distribute_msgs(channel, topic, info)
            
    # NOTE: end() vs stop()
    # end() means everything is done and NO state will be kept.
    # stop() means the process is stopped but the state is still kept.
    def end(self):
        for strat in list(self.strategy_manager.strategies):
            self.strategy_manager.stop(strat, reason='end')
            self.remove_strategy(strat)
        for broker in list(self.brokers.values()):
            broker.stop()
            self.remove_broker(broker.name)
        self._zmq.stop()
        schedule.clear()
        self._is_running = False
        while self._background_thread.is_alive():
            self.logger.debug('waiting for background thread to finish')
            time.sleep(1)
        else:
            self.logger.debug('background thread is finished')
        self._remove_singleton()
