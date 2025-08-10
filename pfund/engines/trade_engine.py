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
    from pfeed._typing import tDataTool
    from pfeed.messaging.zeromq import ZeroMQ
    from pfund.datas.data_time_based import TimeBasedData
    from pfund._typing import DataRangeDict, TradeEngineSettingsDict, tDatabase, ExternalListenersDict

import asyncio
import logging
from threading import Thread

from pfund.engines.base_engine import BaseEngine
from pfund.enums import PFundDataChannel
from pfund import get_config


config = get_config()
logger = logging.getLogger('pfund')


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
        external_listeners: ExternalListenersDict | None=None,
        # TODO: move inside settings?
        df_min_rows: int=1_000,
        df_max_rows: int=3_000,
    ):  
        from pfeed.engine import DataEngine
        from pfund.engines.trade_engine_settings import TradeEngineSettings
        super().__init__(
            env=env,
            name=name,
            data_tool=data_tool,
            data_range=data_range,
            database=database,
            settings=TradeEngineSettings(**(settings or {})),
            external_listeners=external_listeners,
        )
        # TODO:
        # self.DataTool.set_min_rows(df_min_rows)
        # self.DataTool.set_max_rows(df_max_rows)

        self._data_engine = DataEngine(
            env=self._env,
            data_tool=self._data_tool,
            use_ray=not self.is_wasm(),
            use_deltalake=config.use_deltalake
        )
        self._data_engine_thread: Thread | None = None
        self._data_engine_loop: asyncio.AbstractEventLoop | None = None
        self._data_engine_task: asyncio.Task | None = None
        self._worker: ZeroMQ | None = None
        if not self.is_wasm():
            self._setup_data_engine()
    
    @property
    def data_engine(self):
        return self._data_engine
    
    def _setup_data_engine(self):
        from pfeed.messaging.zeromq import ZeroMQ
        sender_name = "data_engine"
        self._data_engine._setup_messaging(
            zmq_url=self._settings.zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL),
            zmq_sender_port=self._settings.zmq_ports.get('data_engine', None),
            # NOTE: zmq_receiver_port is not expected to be set manually
            # zmq_receiver_port=...
        )
        data_engine_zmq = self._data_engine._msg_queue
        data_engine_port = data_engine_zmq.get_ports_in_use(data_engine_zmq.sender)[0]
        cls = self.__class__
        cls._settings.zmq_ports.update({ sender_name: data_engine_port })
    
    def _setup_worker(self):
        import zmq
        from pfeed.messaging.zeromq import ZeroMQ
        # pull from components, e.g. orders
        self._worker = ZeroMQ(name=self.name+"_worker", logger=logger, receiver_type=zmq.PULL)
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
            logger.debug(f"zmq worker connected to {zmq_name} at {zmq_url}:{zmq_port}")
    
    def gather(self):
        super().gather()
        if self._is_gathered:
            return
        for strategy in self.strategies.values():
            datas: list[TimeBasedData] = strategy._get_datas_in_use()
            for data in datas:
                if data.is_resamplee():
                    continue
                self._data_engine \
                .add_feed(data.source, data.category) \
                .stream(
                    product=str(data.product.basis),
                    resolution=repr(data.resolution),
                    **data.product.specs
                )
                # NOTE: load(to_storage=...) is not called here so that users can manually call
                # gather() and add transform() to the feeds in data engine.
    
    def run(self, **ray_kwargs):
        '''
        Args:
            ray_kwargs: keyword arguments passed to pfeed's data engine run() method
                'num_cpus' specifies the number of Ray workers to create in the data engine
                if not specified, all available system CPUs will be used
        '''
        super().run()
        if not self.is_wasm():
            self._setup_worker()
            # NOTE: need to init ray in the main thread to avoid "SIGTERM handler is not set because current thread is not the main thread"
            import ray
            if not ray.is_initialized():
                ray.init(**ray_kwargs)
            self._run_data_engine()
            while self.is_running():
                try:
                    # TODO: receive positions, balances etc.
                    if msg := self._proxy.recv():
                        channel, topic, data, msg_ts = msg
                        if channel == PFundDataChannel.logging:
                            log_level: str = topic
                            log_level: int = logging._nameToLevel.get(log_level.upper(), logging.DEBUG)
                            logger.log(log_level, f'{data}')
                        else:
                            logger.debug(f'{channel} {topic} {data} {msg_ts}')
                    # TODO: receive components orders
                    if msg := self._worker.recv():
                        pass
                except Exception:
                    logger.exception(f"Exception in {self.name} run():")
                except KeyboardInterrupt:
                    logger.warning(f'KeyboardInterrupt received, ending {self.name}')
                    break
            if self.is_running():
                self.end()
            if ray.is_initialized():
                ray.shutdown()
        else:
            # TODO: get msg from data engine
            msg = ...
            for strategy in self.strategies.values():
                strategy.databoy._collect(msg)
    
    def _run_data_engine(self):
        def _run():
            self._data_engine_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._data_engine_loop)
            self._data_engine_task = self._data_engine_loop.create_task(self._data_engine.run_async())
            try:
                self._data_engine_loop.run_until_complete(self._data_engine_task)
            except Exception:
                logger.exception("Exception in data engine thread:")
            finally:
                self._data_engine_loop.close()
                self._data_engine_loop = None
                self._data_engine_task = None
        
        # add storage to feeds in data engine
        for feed in self._data_engine.feeds:
            dataflow = feed.streaming_dataflows[0]
            if dataflow.sink is None:
                feed.load(to_storage=config.storage)
        
        self._data_engine_thread = Thread(target=_run, daemon=True)
        self._data_engine_thread.start()
    
    def _end_data_engine(self):
        logger.debug(f'{self.name} ending data engine')
        if self._data_engine_task and self._data_engine_loop and not self._data_engine_task.done():
            self._data_engine_loop.call_soon_threadsafe(self._data_engine_task.cancel)
        
    def end(self):
        super().end()
        self._end_data_engine()
        if self._data_engine_thread:
            logger.debug(f"{self.name} waiting for data engine thread to finish")
            self._data_engine_thread.join(timeout=10)
            if self._data_engine_thread.is_alive():
                logger.debug(f"{self.name} data engine thread is still running after timeout")
            else:
                logger.debug(f"{self.name} data engine thread finished")
