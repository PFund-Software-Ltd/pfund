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

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from sklearn.model_selection._split import BaseCrossValidator
    from pfeed.enums import DataSource
    from pfeed.typing import tDATA_TOOL
    from pfund.enums import TradFiBroker
    from pfund.typing import DataRangeDict, DatasetSplitsDict, TradeEngineSettingsDict

from apscheduler.schedulers.background import BackgroundScheduler

from pfund.engines.base_engine import BaseEngine
from pfund.brokers.broker_base import BaseBroker
from pfund.utils.utils import flatten_dict, is_port_in_use


class TradeEngine(BaseEngine):
    settings: TradeEngineSettingsDict = {
        'zmq_ports': {},
        'cancel_all_at': {
            'start': True,
            'stop': True,
        },
    }
    scheduler = BackgroundScheduler()
    
    def __init__(
        self,
        env: Literal['SANDBOX', 'PAPER', 'LIVE']='SANDBOX',
        data_tool: tDATA_TOOL='polars',
        data_range: str | DataRangeDict='ytd',
        dataset_splits: int | DatasetSplitsDict | BaseCrossValidator=721,
        use_ray: bool=False,
        use_duckdb: bool=False,
        settings: TradeEngineSettingsDict | None=None,
        df_min_rows: int=1_000,
        df_max_rows: int=3_000,
        # TODO: handle "broker_data_source", e.g. {'IB': 'DATABENTO'}
        broker_data_source: dict[TradFiBroker, DataSource] | None=None,
    ):
        from pfund.zeromq import ZeroMQ

        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            super().__init__(
                env=env,
                data_tool=data_tool, 
                data_range=data_range, 
                dataset_splits=dataset_splits,
                use_ray=use_ray,
                use_duckdb=use_duckdb,
                settings=settings,
            )
            self._is_running = True
            self._zmq = ZeroMQ('engine')
            self.DataTool.set_min_rows(df_min_rows)
            self.DataTool.set_max_rows(df_max_rows)
            self.broker_data_source = broker_data_source

    @classmethod
    def assign_cpus(cls, name) -> list:
        if 'cpu_affinity' in cls.settings and name in cls.settings['cpu_affinity']:
            assigned_cpus = cls.settings['cpu_affinity'][name]
        else:
            assigned_cpus = []
        if not isinstance(assigned_cpus, list):
            assigned_cpus = [assigned_cpus]
        return assigned_cpus
    
    def _assign_zmq_ports(self, zmq_port: int=5557) -> dict:
        _assigned_ports = []
        def _is_port_available(_port):
            _is_port_assigned = (_port in _assigned_ports)
            if is_port_in_use(_port) or _is_port_assigned:
                return False
            else:
                _assigned_ports.append(_port)
                return True
        def _get_port(start_port=None):
            _port = start_port or zmq_port
            if _is_port_available(_port):
                return _port
            else:
                return _get_port(start_port=_port+1)
        self.settings['zmq_ports']['engine'] = _get_port()
        for broker in self.brokers.values():
            if broker.name == 'CRYPTO':
                for exchange in broker.exchanges.values():
                    self.settings['zmq_ports'][exchange.name] = {'rest_api': _get_port()}
                    if not exchange.use_separate_private_ws_url():
                        self.settings['zmq_ports'][exchange.name]['ws_api'] = _get_port()
                    else:
                        self.settings['zmq_ports'][exchange.name]['ws_api'] = {'public': {}, 'private': {}}
                        ws_servers = exchange.get_ws_servers()
                        for ws_server in ws_servers:
                            self.settings['zmq_ports'][exchange.name]['ws_api']['public'][ws_server] = _get_port()
                        for acc in exchange.accounts.keys():
                            self.settings['zmq_ports'][exchange.name]['ws_api']['private'][acc] = _get_port()
            else:
                self.settings['zmq_ports'][broker.name] = _get_port()
        for strategy in self.strategy_manager.strategies.values():
            # FIXME:
            if self._use_ray:
                self.settings['zmq_ports'][strategy.name] = _get_port()
        self.logger.debug(f"{self.settings['zmq_ports']=}")

    def _start_scheduler(self):
        '''start scheduler for background tasks'''
        self.scheduler.add_job(
            self._ping_processes, 'interval', 
            seconds=self._PROCESS_NO_PONG_TOLERANCE_IN_SECONDS // 3
        )
        self.scheduler.add_job(
            self._check_processes, 'interval', 
            seconds=self._PROCESS_NO_PONG_TOLERANCE_IN_SECONDS
        )
        for broker in self.brokers.values():
            broker.schedule_jobs(self.scheduler)
        self.scheduler.start()
    
    def _ping_processes(self):
        self._zmq.send(0, 0, ('engine', 'ping',))

    def _check_processes(self):
        for broker in self.brokers.values():
            connection_manager = broker.cm
            trading_venues = connection_manager.get_trading_venues()
            if reconnect_trading_venues := [trading_venue for trading_venue in trading_venues if not connection_manager.is_process_healthy(trading_venue)]:
                connection_manager.reconnect(reconnect_trading_venues, reason='process not responding')
        if restart_strats := [strat for strat, strategy in self.strategy_manager.strategies.items() if self._use_ray and not self.strategy_manager.is_process_healthy(strat)]:
            self.strategy_manager.restart(restart_strats, reason='process not responding')

    def add_broker(self, bkr: str) -> BaseBroker:
        bkr = bkr.upper()
        if bkr in self.brokers:
            return self.get_broker(bkr)
        Broker = self.get_Broker(bkr)
        broker = Broker(self.env)
        self.brokers[bkr] = broker
        self.logger.debug(f'added {bkr=}')
        return broker
    
    def run(self, zmq_port: int=5557):
        self._assign_zmq_ports(zmq_port=zmq_port)
        self._zmq.start(
            logger=self.logger,
            send_port=self.settings['zmq_ports']['engine'],
            recv_ports=[port for k, port in flatten_dict(self.settings['zmq_ports']).items() if k != 'engine']
        )

        for broker in self.brokers.values():
            broker.start(zmq=self._zmq)

        self.strategy_manager.start()
        
        self._start_scheduler()

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
    
    # TODO: implement pause(), resume(), restart()?
    # def pause(self):
    #     """Pause execution but keep the state."""

    # def resume(self):
    #     """Resume from where it was paused."""

    # def restart(self):
    #     """Stop and restart (soft reset), Resets only part of the state."""
        
    def end(self):
        """Stop and clear all state (hard stop)."""
        for strat in list(self.strategy_manager.strategies):
            self.strategy_manager.stop(strat, reason='end')
            self.remove_strategy(strat)
        for broker in list(self.brokers.values()):
            broker.stop()
            self.remove_broker(broker.name)
        self._zmq.stop()
        self.scheduler.shutdown()
        self._is_running = False
        self._remove_singleton()
