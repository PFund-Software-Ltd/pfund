# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportUnnecessaryComparison=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from logging import Logger
    from pfeed.streaming.zeromq import ZeroMQ
    from pfeed.streaming.streaming_message import StreamingMessage
    from pfund.datas.data_market import MarketData
    from pfund.datas.stores.base_data_store import BaseDataStore
    from pfund.datas.data_bar import BarData
    from pfund.typing import (
        ComponentName,
        Component,
    )

from threading import Thread

import narwhals as nw

from pfeed.enums import DataCategory
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
        # REVIEW: currently all components use ZeroMQ and a thread to run _collect()
        # including even the local ones. if theres any performance issue, 
        # consider disabling using ZeroMQ for local components
        self._zmq_thread: Thread | None = None
    
    @property
    def logger(self) -> Logger:
        return self._component.logger
    
    @property
    def name(self) -> ComponentName:
        return self._component.name
    
    def is_using_zmq(self) -> bool:
        return self._data_zmq is not None and self._signals_zmq is not None

    def _create_data_store(self, category: DataCategory) -> BaseDataStore:
        from pfund.datas.stores.market_data_store import MarketDataStore
        if category == DataCategory.MARKET_DATA:
            return MarketDataStore(self)
        else:
            raise NotImplementedError(f'{category} is not supported')
    
    def get_data_store(self, category: DataCategory | str) -> BaseDataStore:
        category = DataCategory[category.upper()]
        if category in self._data_stores:
            return self._data_stores[category]
        else:
            data_store = self._create_data_store(category)
            self._data_stores[category] = data_store
            return data_store
    
    # FIXME: move to market_data_store
    def _flush_stale_bar(self, data: BarData):
        if data.is_closed():
            self._deliver(data)
    
    # FIXME: move to market_data_store
    def schedule_jobs(self, scheduler: BackgroundScheduler):
        for data, timeout in self.market_data_store._stale_bar_timeouts.items():
            scheduler.add_job(lambda: self._flush_stale_bar(data), 'interval', seconds=timeout)
    
    def _setup_messaging(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ

        if self._data_zmq or self._signals_zmq:
            self.logger.debug(f'{self.name} messaging already setup')
            return
        
        settings = self._component.settings
        zmq_urls = settings.zmq_urls
        zmq_ports = settings.zmq_ports
        
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
        
        zmq_urls.update({
            component_name: component_zmq_url,
        })
        zmq_ports.update({
            signals_zmq_name: signals_zmq_port,
            data_zmq_name: data_zmq_port,
        })
    
    def _subscribe(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ, ZeroMQDataChannel
        from pfund.datas.data_market import MarketData

        if self._data_zmq.get_addresses_in_use(self._data_zmq.receiver) or self._signals_zmq.get_addresses_in_use(self._signals_zmq.receiver):
            self.logger.debug(f'{self.name} already subscribed')
            return
        
        engine_name = self._component.context.name
        settings = self._component.settings
        zmq_urls = settings.zmq_urls
        zmq_ports = settings.zmq_ports
        # subscribe to trade engine (proxy) order updates and data engine's data
        for zmq_name in [engine_name, 'data_engine']:
            if zmq_name not in zmq_urls:
                continue
            zmq_url = zmq_urls[zmq_name]
            zmq_port = zmq_ports[zmq_name]
            self._data_zmq.connect(
                socket=self._data_zmq.receiver,
                port=zmq_port,
                url=zmq_url,
            )
            self.logger.debug(f'{self._data_zmq.name} connected to {zmq_name} at {zmq_url}:{zmq_port}')
        
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
        for data in self._component.get_datas():
            if isinstance(data, MarketData):
                zmq_channel = ZeroMQDataChannel.create_market_data_channel(
                    data_source=data.source,
                    product=data.product,
                    resolution=data.resolution,
                )
                self._data_zmq.receiver.setsockopt(zmq.SUBSCRIBE, zmq_channel.encode())
            else:
                raise NotImplementedError(f'Unhandled data type: {type(data)}')

        for component in self._component.get_components():
            component_zmq_url = settings.zmq_urls.get(component.name, ZeroMQ.DEFAULT_URL)
            component_zmq_port = zmq_ports.get(component.name, None)
            self._signals_zmq.connect(
                socket=self._signals_zmq.receiver,
                port=component_zmq_port,
                url=component_zmq_url,
            )
            self.logger.debug(f'{self._signals_zmq.name} connected to {component.name} at {component_zmq_url}:{component_zmq_port}')
    
    # FIXME
    def pong(self):
        """Pongs back to Engine's ping to show that it is alive"""
        zmq_msg = (0, 0, (self.strat,))
        self._zmq.send(*zmq_msg, receiver='engine')

    def start(self):
        if self._data_zmq or self._signals_zmq:
            self._zmq_thread = Thread(target=self._collect, daemon=True)
            self._zmq_thread.start()

    def stop(self):
        if self._zmq_thread and self._zmq_thread.is_alive():
            self.logger.debug(f"{self.name} waiting for data thread to finish")
            self._zmq_thread.join(timeout=10)  # Blocks until thread finishes
            self.logger.debug(f"{self.name} data thread finished (alive={self._zmq_thread.is_alive()})")
    
    def _update_data_store(self, msg: StreamingMessage):
        from msgspec import structs
        
        data_store = self.get_data_store(msg.data_category)
        update = structs.asdict(msg)
        
        if msg.data_category == DataCategory.MARKET_DATA:
            if msg.is_quote():
                data_store.update_quote(update)
            elif msg.is_tick():
                data_store.update_tick(update)
            elif msg.is_bar():
                data_store.update_bar(update)
            else:
                raise ValueError(f'Unhandled market data message: {msg}')
        else:
            raise NotImplementedError(f'Unhandled data category: {msg.data_category}')
    
    def _collect(self, msg: StreamingMessage | None=None):
        '''
        Args:
            msg: StreamingMessage, only provided when data is not being sent via zeromq
        '''
        # when not using zeromq (guaranteed ALL components are local components)
        if msg is not None:
            for component in self._component.get_components():
                component.databoy._collect(msg=msg)
            self._update_data_store(msg)
        # when using zeromq (there could be some local and remote components, but both use zeromq to receive data anyways)
        else:
            while self._component.is_running():
                if msg_tuple := self._data_zmq.recv():
                    channel, topic, msg, msg_ts = msg_tuple
                    
                    # TEMP
                    # print('databoy data_zmq recv:', channel, topic, msg, msg_ts)
                    
                    self._update_data_store(msg)
                # TODO:
                if msg_tuple := self._signals_zmq.recv():
                    pass
                
                # TODO: check if signals are ready, if yes, call back on trade(X)
    
    # TODO: should receive both signals_dict and signals_df
    def _wait_for_children_signals(self, timeout: float = 10.0):
        '''Waits for all children's signals via signals_zmq before delivering data.
        Only used in ZMQ mode when the component has children.
        zmq.recv() releases the GIL, so other threads (children) can compute their signals.
        '''
        import time

        pending = set(c.name for c in self._component.get_components())
        deadline = time.monotonic() + timeout
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.logger.warning(
                    f'{self.name} timed out waiting for children signals: {pending}'
                )
                break
            if msg_tuple := self._signals_zmq.recv(timeout=int(remaining * 1000)):
                channel, topic, signal, msg_ts = msg_tuple
                # TODO: store the signal and extract child name from msg
                child_name = topic
                pending.discard(child_name)

    def _deliver(self, data: MarketData):
        '''Deliver data to the component'''
        if data.category == DataCategory.MARKET_DATA:
            if data.is_quote():
                self._component._on_quote(data)
            elif data.is_tick():
                self._component._on_tick(data)
            elif data.is_bar():
                if data.is_closed():
                    self._data_stores[data.category].update_df(data)
                self._component._on_bar(data)
        else:
            raise NotImplementedError(f'Unhandled data type: {type(data)}')