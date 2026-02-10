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
from typing import TYPE_CHECKING, Literal, cast
if TYPE_CHECKING:
    from pfeed.streaming.zeromq import ZeroMQ
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.datas.resolution import Resolution
    from pfund.engines.engine_context import DataRangeDict

import logging

from pfund.engines.base_engine import BaseEngine
from pfund.enums import Environment, PFundDataChannel


# TODO: SANDBOX env + backtest data = replaying
class TradeEngine(BaseEngine):
    def __init__(
        self,
        *,
        env: Environment | Literal['SANDBOX', 'PAPER', 'LIVE']=Environment.SANDBOX,
        data_range: str | Resolution | DataRangeDict | Literal['ytd']='ytd',
    ):  
        super().__init__(env=Environment[env.upper()], data_range=data_range)
        self._proxy: ZeroMQ | None = None
        self._worker: ZeroMQ | None = None
        self._setup_proxy()
    
    @property
    def settings(self) -> TradeEngineSettings:
        return cast("TradeEngineSettings", self._context.settings)
    
    def _setup_proxy(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ
        # FIXME: remove zmq.XPUB, use mtflow's ws_client to emit to mtflow's websocket server (external listeners)
        self._proxy = ZeroMQ(
            name=self.name+"_proxy",
            logger=self._logger,
            io_threads=2,
            sender_type=zmq.XPUB,  # publish to external listeners
            receiver_type=zmq.XSUB,  # subscribe to data engine, component's logs (if using ray) etc.
        )
        sender_name = "proxy"
        zmq_ports = self.settings.zmq_ports
        engine_zmq_url = self.settings.zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL)
        self._proxy.bind(
            socket=self._proxy.sender,
            port=zmq_ports.get(sender_name, None),
            url=engine_zmq_url,
        )
        proxy_zmq_port= self._proxy.get_ports_in_use(self._proxy.sender)[0]
        self._logger.debug(f"zmq proxy binded to {engine_zmq_url}:{proxy_zmq_port}")
        for zmq_name, zmq_port in zmq_ports.items():
            if zmq_name == 'data_engine' or zmq_name.endswith("_logger"):
                self._proxy.connect(
                    socket=self._proxy.receiver,
                    port=zmq_port,
                    url=engine_zmq_url,
                )
                self._logger.debug(f"zmq proxy connected to {zmq_name} at {engine_zmq_url}:{zmq_port}")
        self.settings.zmq_ports.update({ sender_name: proxy_zmq_port })
    
    def _setup_worker(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ
        # pull from components, e.g. orders
        self._worker = ZeroMQ(name=self.name+"_worker", logger=self._logger, receiver_type=zmq.PULL)
        for zmq_name, zmq_port in self.settings.zmq_ports.items():
            if zmq_name in ['proxy', 'data_engine'] or zmq_name.endswith("_logger"):
                continue
            if zmq_name.endswith("_data"):
                component_name = zmq_name.replace("_data", "")
            else:
                component_name = zmq_name
            zmq_url = self.settings.zmq_urls.get(component_name, ZeroMQ.DEFAULT_URL)
            self._worker.connect(
                socket=self._worker.receiver,
                port=zmq_port,
                url=zmq_url,
            )
            self._logger.debug(f"zmq worker connected to {zmq_name} at {zmq_url}:{zmq_port}")
    
    def _gather(self):
        super()._gather()
        if not self._is_gathered:
            self._setup_worker()
            for strategy in self.strategies.values():
                # updates zmq ports in settings
                self.settings.zmq_ports.update(strategy._get_zmq_ports_in_use())
    
    def run(self):
        super().run()
        # NOTE: need to init ray in the main thread to avoid "SIGTERM handler is not set because current thread is not the main thread"
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
    
    def end(self):
        super().end()
        if self._proxy:
            self._proxy.terminate()
        if self._worker:
            self._worker.terminate()
