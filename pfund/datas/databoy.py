from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from logging import Logger
    from pfeed.streaming.zeromq import ZeroMQ
    from pfund.datas.stores.market_data_store import MarketDataStore
    from pfund.datas.data_market import MarketData
    from pfund.datas.stores.base_data_store import BaseDataStore
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_bar import BarData
    from pfund.engines.engine_context import EngineContext
    from pfund.typing import (
        ComponentName, 
        Component, 
        ZeroMQSenderName, 
    )

from threading import Thread

from pfeed.enums import DataCategory
from pfund.entities.products.product_base import BaseProduct

from pfund.datas.resolution import Resolution
from pfund.enums import PublicDataChannel, PrivateDataChannel


class DataBoy:
    def __init__(self, component: Component):
        self._component: Component = component
        self._data_stores: dict[DataCategory, BaseDataStore] = {}
        self._data_zmq: ZeroMQ | None = None
        self._signals_zmq: ZeroMQ | None = None
        # TODO: save data signatures properly, data_signatures should be a set
        # TODO: add data_config (dict form) to data_signatures
        self._data_signatures = []
        # REVIEW: currently all non-WASM components use ZeroMQ and a thread to run _collect()
        # including even the local ones. if theres any performance issue, 
        # consider disabling using ZeroMQ for local components
        self._zmq_thread: Thread | None = None
        self._zmq_ports_in_use: dict[ZeroMQSenderName, int] = {}

    @property
    def logger(self) -> Logger:
        return self._component.logger
    
    @property
    def name(self) -> ComponentName:
        return self._component.name
    
    @property
    def components(self) -> list[Component]:
        return self._component.components
    
    @property
    def consumers(self) -> list[Component]:
        return self._component.consumers
    
    @property
    def context(self) -> EngineContext:
        return self._component.context
    
    @property
    def data_stores(self) -> dict[DataCategory, BaseDataStore]:
        return self._data_stores
    
    @property
    def market_data_store(self) -> MarketDataStore:
        return self.get_data_store(DataCategory.MARKET_DATA)
    
    def is_remote(self) -> bool:
        return self._component.is_remote()

    def _create_data_store(self, category: DataCategory) -> MarketDataStore:
        if category == DataCategory.MARKET_DATA:
            return MarketDataStore(self.context)
        else:
            raise ValueError(f'{category} is not supported')
        
    def get_data_store(self, category: DataCategory | str) -> MarketDataStore:
        category = DataCategory[category.upper()]
        if category in self._data_stores:
            return self._data_stores[category]
        else:
            data_store = self._create_data_store(category)
            self._data_stores[category] = data_store
            return data_store

    def get_datas(self) -> list[BaseData]:
        datas = []
        for data_store in self._data_stores.values():
            datas.extend(data_store.get_datas())
        return datas
    
    def _flush_stale_bar(self, data: BarData):
        if data.is_closed():
            self._deliver(data)
    
    def schedule_jobs(self, scheduler: BackgroundScheduler):
        for data, timeout in self.market_data_store.stale_bar_timeouts.items():
            scheduler.add_job(lambda: self._flush_stale_bar(data), 'interval', seconds=timeout)
    
    def _setup_messaging(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ
        
        settings = self._component.settings
        zmq_urls = settings.zmq_urls
        zmq_ports = settings.zmq_ports
        self._update_zmq_ports_in_use(zmq_ports)
        
        component_name = self.name
        component_zmq_url = zmq_urls.get(component_name, ZeroMQ.DEFAULT_URL)

        data_zmq_name = component_name + '_data'
        self._data_zmq = ZeroMQ(
            name=data_zmq_name,
            logger=self.logger,
            sender_type=zmq.PUSH,  # send component created data (e.g. orders) to trade engine
            receiver_type=zmq.SUB,  # receive data from data engine, order updates from trade engine
        )
        self._data_zmq.bind(
            socket=self._data_zmq.sender,
            port=zmq_ports.get(data_zmq_name, None),
            url=component_zmq_url,
        )
        data_zmq_port = self._data_zmq.get_ports_in_use(self._data_zmq.sender)[0]
        
        signals_zmq_name = component_name
        self._signals_zmq = ZeroMQ(
            name=signals_zmq_name,
            logger=self.logger,
            sender_type=zmq.PUB,  # publish signals to other consumers
            receiver_type=zmq.SUB,  # subscribe to signals from other components
        )
        self._signals_zmq.bind(
            socket=self._signals_zmq.sender,
            port=zmq_ports.get(signals_zmq_name, None),
            url=component_zmq_url,
        )
        signals_zmq_port = self._signals_zmq.get_ports_in_use(self._signals_zmq.sender)[0]
        
        self._update_zmq_ports_in_use({
            data_zmq_name: data_zmq_port,
            signals_zmq_name: signals_zmq_port
        })
    
    def _get_zmq_ports_in_use(self) -> dict[ZeroMQSenderName, int]:
        '''Gets ALL zmq ports in use even the ones used in components'''
        for component in self.components:
            self._zmq_ports_in_use.update(component._get_zmq_ports_in_use())
        return self._zmq_ports_in_use

    def _update_zmq_ports_in_use(self, zmq_ports: dict[ZeroMQSenderName, int]):
        self._zmq_ports_in_use.update(zmq_ports)
    
    def _subscribe(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ, ZeroMQDataChannel
        zmq_ports = self._get_zmq_ports_in_use()
        engine_name = self._component._context.name
        settings = self._component.settings
        engine_zmq_url = settings.zmq_urls.get(engine_name, ZeroMQ.DEFAULT_URL)
        # subscribe to proxy's order updates and data engine's data
        for zmq_name in ['proxy', 'data_engine']:
            zmq_port = zmq_ports.get(zmq_name, None)
            self._data_zmq.connect(
                socket=self._data_zmq.receiver,
                port=zmq_port,
                url=engine_zmq_url,
            )
            self.logger.debug(f'{self._data_zmq.name} connected to {zmq_name} at {engine_zmq_url}:{zmq_port}')
        
        # subscribe to private channels: positions, balances, orders, etc.
        if self._component.is_strategy():
            accounts = list(self._component.accounts.values())
            channels = list(PrivateDataChannel.__members__)
            for account, channel in zip(accounts, channels):
                zmq_channel: str = ZeroMQDataChannel.create_private_channel(
                    account=account,
                    channel=channel,
                )
                self._data_zmq.receiver.setsockopt(zmq.SUBSCRIBE, zmq_channel.encode())
        
        # subscribe to data channels: quote, tick, bar, etc.
        for data in self.get_datas():
            if isinstance(data, MarketData):
                zmq_channel = ZeroMQDataChannel.create_market_data_channel(
                    data_source=data.source,
                    product=data.product,
                    resolution=data.resolution,
                )
                self._data_zmq.receiver.setsockopt(zmq.SUBSCRIBE, zmq_channel.encode())
            else:
                raise NotImplementedError(f'Unhandled data type: {type(data)}')

        for component in self.components:
            component_name = component.name
            component_zmq_url = settings.zmq_urls.get(component_name, ZeroMQ.DEFAULT_URL)
            component_zmq_port = zmq_ports.get(component_name, None)
            self._signals_zmq.connect(
                socket=self._signals_zmq.receiver,
                port=component_zmq_port,
                url=component_zmq_url,
            )
            self.logger.debug(f'{self._signals_zmq.name} connected to {component_name} at {component_zmq_url}:{component_zmq_port}')
    
    # FIXME
    def pong(self):
        """Pongs back to Engine's ping to show that it is alive"""
        zmq_msg = (0, 0, (self.strat,))
        self._zmq.send(*zmq_msg, receiver='engine')

    def start(self):
        # set the ZMQPubHandler's receiver ready to flush the buffered log messages
        if self.is_remote():
            self.logger.handlers[0].set_receiver_ready()
        if self._data_zmq or self._signals_zmq:
            self._zmq_thread = Thread(target=self._collect, daemon=True)
            self._zmq_thread.start()

    def stop(self):
        if self._zmq_thread and self._zmq_thread.is_alive():
            self.logger.debug(f"{self.name} waiting for data thread to finish")
            self._zmq_thread.join()  # Blocks until thread finishes
            self.logger.debug(f"{self.name} data thread finished")
    
    # TODO:
    def _send_signal(self, signal):
        # self._signals_zmq.send(signal)
        pass
    
    def _collect(self):
        from msgspec import structs
        while self._component.is_running():
            if msg_tuple := self._data_zmq.recv():
                # TODO: how to know which data store to use from msg_tuple?
                channel, topic, msg, msg_ts = msg_tuple
                
                # TEMP
                print('databoy data_zmq recv:', channel, topic, msg, msg_ts)
                
                product: BaseProduct = self._component.get_product(msg.product)
                resolution = Resolution(msg.resolution)
                data: MarketData = self.market_data_store.get_data(product.name, resolution)
                update = structs.asdict(msg)
                if topic == PublicDataChannel.orderbook:
                    self.market_data_store.update_quote(data, update)
                    self._deliver(data)
                elif topic == PublicDataChannel.tradebook:
                    self.market_data_store.update_tick(data, update)
                    self._deliver(data)
                elif topic == PublicDataChannel.candlestick:
                    # deliver the closed bar before update() clears it for the next bar
                    if not update['is_incremental']:
                        self.market_data_store.update_bar(data, update)
                        self._deliver(data)
                    else:
                        if data.is_closed(now=update['ts'] or update['msg_ts']):
                            self._deliver(data)
                        self.market_data_store.update_bar(data, update)
                else:
                    raise NotImplementedError(f'{topic=} is not supported')
            # TODO:
            if msg_tuple := self._signals_zmq.recv():
                pass
            
            # TODO: check if signals are ready, if yes, call back on trade(X)
    
    def _deliver(self, data: MarketData):
        '''Deliver data to the component'''
        if self._component.is_remote():
            # TODO
            self._send_signal(...)
        else:
            if data.resolution.is_quote():
                self._component._on_quote(data)
                for data_resamplee in data.get_resamplees():
                    self._deliver(data_resamplee)
            elif data.resolution.is_tick():
                self._component._on_tick(data)
                for data_resamplee in data.get_resamplees():
                    self._deliver(data_resamplee)
            elif data.resolution.is_bar():
                self._component._on_bar(data)
                for data_resamplee in data.get_resamplees():
                    if data_resamplee.is_closed(now=data.end_ts) and not data_resamplee.skip_first_bar():
                        self._deliver(data_resamplee)
