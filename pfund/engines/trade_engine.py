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
    from pfeed.typing import tDataTool
    from pfeed.messaging.zeromq import ZeroMQ
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.typing import DataRangeDict, TradeEngineSettingsDict, tDatabase

import logging

from pfund.engines.base_engine import BaseEngine
from pfund.enums import PFundDataChannel
from pfund import get_config


config = get_config()


class TradeEngine(BaseEngine):
    def __init__(
        self,
        *,
        env: Literal['SANDBOX', 'PAPER', 'LIVE'],
        name: str='',
        data_tool: tDataTool='polars',
        data_range: str | DataRangeDict='ytd',
        database: tDatabase | None=None,
        settings: TradeEngineSettingsDict | None=None,
        # TODO: move inside settings?
        df_min_rows: int=1_000,
        df_max_rows: int=3_000,
    ):  
        from pfund.engines.trade_engine_settings import TradeEngineSettings
        super().__init__(
            env=env,
            name=name,
            data_tool=data_tool,
            data_range=data_range,
            database=database,
            settings=TradeEngineSettings(**(settings or {})),
        )
        # TODO:
        # self.DataTool.set_min_rows(df_min_rows)
        # self.DataTool.set_max_rows(df_max_rows)
        self._worker: ZeroMQ | None = None
    
    def _setup_worker(self):
        import zmq
        from pfeed.messaging.zeromq import ZeroMQ
        # pull from components, e.g. orders
        self._worker = ZeroMQ(name=self.name+"_worker", logger=self._logger, receiver_type=zmq.PULL)
        for zmq_name, zmq_port in self._settings.zmq_ports.items():
            if zmq_name in ['proxy', 'data_engine'] or zmq_name.endswith("_logger"):
                continue
            if zmq_name.endswith("_data"):
                component_name = zmq_name.replace("_data", "")
            else:
                component_name = zmq_name
            zmq_url = self._settings.zmq_urls.get(component_name, ZeroMQ.DEFAULT_URL)
            self._worker.connect(
                socket=self._worker.receiver,
                port=zmq_port,
                url=zmq_url,
            )
            self._logger.debug(f"zmq worker connected to {zmq_name} at {zmq_url}:{zmq_port}")
    
    def gather(self):
        if self._is_gathered:
            return
        super().gather()
        self._setup_worker()
        for strategy in self.strategies.values():
            datas: list[TimeBasedData] = strategy._get_datas_in_use()
            for data in datas:
                if data.is_resamplee():
                    continue
        self._is_gathered = True
    
    def run(self, **ray_kwargs):
        '''
        Args:
            ray_kwargs: keyword arguments for ray.init()
        '''
        super().run()
        # NOTE: need to init ray in the main thread to avoid "SIGTERM handler is not set because current thread is not the main thread"
        import ray
        if not ray.is_initialized():
            ray.init(**ray_kwargs)
        while self.is_running():
            try:
                # TODO: receive positions, balances etc.
                if msg := self._proxy.recv():
                    channel, topic, data, msg_ts = msg
                    if channel == PFundDataChannel.logging:
                        log_level: str = topic
                        log_level: int = logging._nameToLevel.get(log_level.upper(), logging.DEBUG)
                        self._logger.log(log_level, f'{data}')
                    else:
                        self._logger.debug(f'{channel} {topic} {data} {msg_ts}')
                # TODO: receive components orders
                if msg := self._worker.recv():
                    pass
            except Exception:
                self._logger.exception(f"Exception in {self.name} run():")
            except KeyboardInterrupt:
                self._logger.warning(f'KeyboardInterrupt received, ending {self.name}')
                break
        if self.is_running():
            self.end()
        if ray.is_initialized():
            ray.shutdown()
    