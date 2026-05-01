# pyright: reportArgumentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false
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
    from pfeed.streaming.streaming_message import StreamingMessage
    from pfeed.engine import DataEngine
    from pfund.typing import Component
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.datas.resolution import Resolution
    from pfund.engines.engine_context import DataRangeDict

import time
import logging
from threading import Thread

from pfeed.enums import DataCategory
from pfund.engines.base_engine import BaseEngine
from pfund.enums import Environment, PFundDataChannel, RunStage


class TradeEngine(BaseEngine):
    def __init__(
        self,
        *,
        env: Environment | Literal['SANDBOX', 'PAPER', 'LIVE']=Environment.SANDBOX,
        name: str='engine',
        data_range: str | Resolution | DataRangeDict | Literal['ytd']='ytd',
        settings: TradeEngineSettings | None=None,
    ):
        super().__init__(env=Environment[env.upper()], name=name, data_range=data_range, settings=settings)
        self._proxy: ZeroMQ | None = None
        self._worker: ZeroMQ | None = None
        self._data_engine: DataEngine | None = None
        self._zmq_thread: Thread | None = None
        
    @property
    def settings(self) -> TradeEngineSettings:
        return cast("TradeEngineSettings", self._context.settings)
    
    # TODO: include descriptions
    def show_zmq_graph(self):
        try:
            import graphviz
        except ImportError:
            raise ImportError('graphviz is not installed, please install it using `pip install graphviz`')
        # TEMP
        from pprint import pprint
        print('engine.settings.zmq_urls:')
        pprint(self.settings.zmq_urls.to_dict())
        print('engine.settings.zmq_ports:')
        pprint(self.settings.zmq_ports.to_dict())
    
    def _setup_data_engine(self):
        import pfeed as pe
        from pfeed.streaming.zeromq import ZeroMQ
        
        self._data_engine = pe.DataEngine()
        is_using_zmq = self._is_using_zmq()
        
        if is_using_zmq:
            # setup messaging for data engine
            sender_name = "data_engine"
            zmq_url = self.settings.zmq_urls.get(sender_name, ZeroMQ.DEFAULT_URL)
            zmq_port = self.settings.zmq_ports.get(sender_name, None)
            self._data_engine.setup_messaging(
                zmq_url=zmq_url,
                zmq_sender_port=zmq_port,
            )
            data_engine_zmq = self._data_engine._msg_queue
            assert data_engine_zmq is not None, 'data engine zmq is not set'
            self.settings.zmq_urls.update({ sender_name: zmq_url })
            data_engine_port = data_engine_zmq.get_ports_in_use(data_engine_zmq.sender)[0]
            self.settings.zmq_ports.update({ sender_name: data_engine_port })
        
        def _collect_msg_if_not_using_ray(msg: StreamingMessage):
            for strategy in self.strategies.values():
                strategy.databoy._collect(msg=msg)
            return msg
        
        # data engine creates feeds and subscribes to market data to prepare for streaming
        for data in self._get_all_datas():
            if data.category != DataCategory.MARKET_DATA:
                raise NotImplementedError(f"Unhandled data type: {type(data)}")
            if data.is_resamplee():
                continue
            num_stream_workers = data.config.num_stream_workers
            if is_using_zmq and num_stream_workers is None:
                num_stream_workers = 1
                self._logger.debug(f"defaulting {data} num_stream_workers to 1")
            feed = self._data_engine.add_feed(
                data_source=data.source,
                data_category=data.category,
                num_workers=num_stream_workers,
            )
            feed.stream(
                product=str(data.product.basis),
                resolution=repr(data.resolution),
                data_origin=data.origin,
                env=self.env,
                storage_config=data.storage_config,
                **data.product.specs,
            )
            # if not using zmq, data will be sent via transform()
            if not is_using_zmq:
                feed.transform(_collect_msg_if_not_using_ray)

    def _setup_proxy(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ

        self._proxy = ZeroMQ(
            name=self.name+"_proxy",
            logger=self._logger,
            io_threads=2,
            sender_type=zmq.XPUB,  # publish order updates (from websocket), engine states, to components and external listeners
            receiver_type=zmq.SUB,  # subscribe to data engine, component's logs (if using ray) etc.
        )
        zmq_url = self.settings.zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL)
        zmq_port = self.settings.zmq_ports.get(self.name, None)
        self._proxy.bind(
            socket=self._proxy.sender,
            port=zmq_port,
            url=zmq_url,
        )
        self.settings.zmq_urls.update({ self.name: zmq_url })
        zmq_port= self._proxy.get_ports_in_use(self._proxy.sender)[0]
        self.settings.zmq_ports.update({ self.name: zmq_port })
        self._logger.debug(f"{self.name} zmq proxy binded to {zmq_url}:{zmq_port}")

        # proxy connects to data engine and component's ZMQPubHandler (if using ray)
        for zmq_name, zmq_port in self.settings.zmq_ports.items():
            if zmq_name == 'data_engine':
                zmq_url = self.settings.zmq_urls['data_engine']
            else:
                is_component_logger = zmq_name.endswith("_logger")
                if is_component_logger:
                    component_name = zmq_name.replace("_logger", "")
                    zmq_url = self.settings.zmq_urls[component_name]
                else:
                    continue
            self._proxy.connect(
                socket=self._proxy.receiver,
                port=zmq_port,
                url=zmq_url,
            )
            self._logger.debug(f"{self.name} zmq proxy connected to {zmq_name} at {zmq_url}:{zmq_port}")
        # subscribe XSUB to all topics from all connected upstream publishers
        self._proxy.receiver.setsockopt(zmq.SUBSCRIBE, b"")

    def _setup_worker(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ
        # pull from component orders
        self._worker = ZeroMQ(name=self.name+"_worker", logger=self._logger, receiver_type=zmq.PULL)
        for zmq_name, zmq_port in self.settings.zmq_ports.items():
            if zmq_name in [self.name, 'data_engine'] or zmq_name.endswith("_logger"):
                continue
            is_component_data_zmq = zmq_name.endswith("_data")
            if is_component_data_zmq:
                component_name = zmq_name.replace("_data", "")
            else:
                component_name = zmq_name
            zmq_url = self.settings.zmq_urls[component_name]
            self._worker.connect(
                socket=self._worker.receiver,
                port=zmq_port,
                url=zmq_url,
            )
            self._logger.debug(f"{self.name} zmq worker connected to {zmq_name} at {zmq_url}:{zmq_port}")
    
    def _is_using_zmq(self) -> bool:
        """Returns True if any strategy is remote or has any remote component or any data has num_stream_workers
        Conceptually it is equivalent to: if Ray is being used, then ZeroMQ is also being used.
        """
        def _has_any_remote_component(component: Component) -> bool:
            for _component in component.get_components():
                if _component.is_remote() or _has_any_remote_component(_component):
                    return True
            return False
        for strategy in self.strategies.values():
            if strategy.is_remote() or _has_any_remote_component(strategy):
                return True
            for data in strategy.get_datas():
                if data.config.num_stream_workers is not None:
                    return True
        return False
    
    def _run_zmq_loop(self):
        self._setup_proxy()
        for strategy in self.strategies.values():
            strategy._setup_messaging()
        self._setup_worker()
        assert self._proxy is not None, 'proxy is not set'
        assert self._worker is not None, 'worker is not set'

        while self.is_running():
            try:
                if msg := self._proxy.recv():
                    channel, topic, data, msg_ts = msg

                    
                    if channel == PFundDataChannel.logging:
                        log_level: str = topic
                        log_level: int = logging._nameToLevel.get(log_level.upper(), logging.DEBUG)
                        self._logger.log(log_level, f'{data}')
                    else:
                        
                        # TEMP
                        print('proxy recv:', channel, topic, data, msg_ts)
                        
                        self._logger.debug(f'{channel} {topic} {data} {msg_ts}')
                    # TODO: broker._distribute_msgs(channel, topic, data)
                    
                # TODO: receive positions, balances, components orders etc.
                if msg := self._worker.recv():
                    channel, topic, data, msg_ts = msg
                    
                    # TEMP
                    print('worker recv:', channel, topic, data, msg_ts)
                    
                # TODO: publish orders, positions, balances etc. to components
                # self._proxy.send(...)
            except Exception:
                self._logger.exception(f"Exception in {self.name} _run_zmq_loop():")
            except KeyboardInterrupt:
                self._logger.warning('KeyboardInterrupt received, ending ZMQ loop')
                break
        self._proxy.terminate()
        self._worker.terminate()
    
    def run(self):
        try:
            if self.settings.auto_stream:
                self._setup_data_engine()
            if self._is_using_zmq():
                self._zmq_thread = Thread(target=self._run_zmq_loop, daemon=True)
                self._zmq_thread.start()
            super().run()
            if self._data_engine:
                self._data_engine.run()  # blocking call
            else:
                if self._zmq_thread:
                    self._zmq_thread.join()
                else:
                    while self.is_running():
                        time.sleep(0.1)
        except KeyboardInterrupt:
            self._logger.warning(f'KeyboardInterrupt received, ending {self.name}')
        except Exception:
            self._logger.exception(f"Exception in {self.name} run():")
        finally:
            self.end()

    def end(self):
        super().end()
        if self._data_engine:
            self._data_engine.end()
        if self._zmq_thread and self._zmq_thread.is_alive():
            self._logger.debug(f"{self.name} waiting for zmq thread to finish")
            self._zmq_thread.join(timeout=10)
            self._logger.debug(f"{self.name} zmq thread finished (alive={self._zmq_thread.is_alive()})")
