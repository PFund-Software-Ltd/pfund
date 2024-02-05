"""This is an engine used to trade against multiple brokers and cryptocurrency exchanges.

This engine is designed for algorithmic trading and contains all the major 
components at the highest level such as:
    brokers (e.g. Interactive Brokers, Crypto, ...),
        where broker `Crypto` is a fake broker name that includes the actual
        crypto exchanges (e.g. Binance, Bybit, ...)
    strategies (your trading strategies)
In order to communicate with other processes, it uses ZeroMQ as the core 
message queue.

Please refer to #TODO for more examples.
"""
import time
from threading import Thread

import schedule
import psutil

from pfund.data_tools.data_tool_base import DataTool
from pfund.engines.base_engine import BaseEngine
from pfund.brokers.broker_base import BaseBroker
from pfund.const.commons import *
from pfund.utils.utils import flatten_dict
from pfund.zeromq import ZeroMQ
from pfund.config_handler import Config


class TradeEngine(BaseEngine):
    zmq_ports = {}

    def __new__(cls, *, env: str='PAPER', data_tool: DataTool='pandas', zmq_port=5557, config: Config | None=None, **settings):
        if not hasattr(cls, 'zmq_port'):
            assert type(zmq_port) is int, f'{zmq_port=} must be an integer'
            cls._zmq_port = zmq_port
        return super().__new__(cls, env, data_tool=data_tool, config=config, **settings)

    def __init__(self, *, env: str='PAPER', data_tool: DataTool='pandas', zmq_port=5557, config: Config | None=None, **settings):
        super().__init__(env, data_tool=data_tool)
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            self._is_running = True
            self._zmq = ZeroMQ('engine')
            self._background_thread = None

    @classmethod
    def assign_cpus(cls, name) -> list:
        if 'cpu_affinity' in cls.configs and name in cls.configs['cpu_affinity']:
            assigned_cpus = cls.configs['cpu_affinity'][name]
        else:
            assigned_cpus = []
        if not isinstance(assigned_cpus, list):
            assigned_cpus = [assigned_cpus]
        return assigned_cpus
    
    def _assign_zmq_ports(self) -> dict:
        _assigned_ports = []
        def _is_port_available(_port):
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                _is_port_in_use = (s.connect_ex(('localhost', _port)) == 0)
                _is_port_assigned = (_port in _assigned_ports)
                if _is_port_in_use or _is_port_assigned:
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
                    if not exchange.is_use_private_ws_server():
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

    # TODO, should be called by dashboard
    def monitor_cpu_usage(self):
        pcts = psutil.cpu_percent(interval=1, percpu=True)
        for cpu_num in range(len(pcts)):
            pct = pcts[cpu_num]
            if pct > 0:
                self.logger.warning(f'cpu {cpu_num} has {pct}% usage')

    # TODO, in case no more space for logs
    def monitor_memory_usage():
        pass

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
        self.logger.debug(f'background thread started')

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
            lats = []
            if msg := self._zmq.recv():
                channel, topic, info = msg
                # TODO
                if channel == 0:  # from strategy processes to strategy manager
                    self.strategy_manager.handle_msgs(topic, info)
                else:
                    bkr = info[0]
                    broker = self.brokers[bkr]
                    broker.distribute_msgs(channel, topic, info)
            
    def end(self):
        self.strategy_manager.stop()
        for broker in self.brokers.values():
            broker.stop()
        self._zmq.stop()
        schedule.clear()
        self._is_running = False
        while self._background_thread.is_alive():
            self.logger.debug(f'waiting for background thread to finish')
            time.sleep(1)
        else:
            self.logger.debug(f'background thread is finished')
