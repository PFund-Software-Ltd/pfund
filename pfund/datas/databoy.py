# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportUnnecessaryComparison=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
if TYPE_CHECKING:
    from logging import Logger
    from narwhals._native import NativeDataFrame
    from pfeed.streaming.zeromq import ZeroMQ
    from pfund.datas.stores.market_data_store import MarketDataStore
    from pfund.datas.stores.trading_store import TradingStore
    from pfeed.streaming.streaming_message import StreamingMessage
    from pfund.datas.data_market import MarketData
    from pfund.datas.stores.base_data_store import BaseDataStore
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_bar import BarData
    from pfund.engines.engine_context import EngineContext
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
        self._trading_store: TradingStore | None = None
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
    def trading_store(self) -> TradingStore:
        return self.get_trading_store()
    
    @property
    def market_data_store(self) -> MarketDataStore:
        return self.get_data_store(DataCategory.MARKET_DATA)
    
    def is_remote(self) -> bool:
        return self._component.is_remote()
    
    def _create_trading_store(self) -> TradingStore:
        from pfund.datas.stores.trading_store import TradingStore
        return TradingStore(self.context)

    def _create_data_store(self, category: DataCategory) -> MarketDataStore:
        from pfund.datas.stores.market_data_store import MarketDataStore
        if category == DataCategory.MARKET_DATA:
            return MarketDataStore(self)
        else:
            raise ValueError(f'{category} is not supported')
    
    def get_data_store(self, category: DataCategory | str) -> MarketDataStore:
        category = DataCategory[category.upper()]
        if category in self.data_stores:
            return self.data_stores[category]
        else:
            data_store = self._create_data_store(category)
            self.data_stores[category] = data_store
            return data_store

    def get_trading_store(self) -> TradingStore:
        if self._trading_store is None:
            self._trading_store = self._create_trading_store()
        return self._trading_store
    
    def get_datas(self) -> list[BaseData]:
        datas = []
        for data_store in self.data_stores.values():
            datas.extend(data_store.get_datas())
        return datas
    
    def get_df(
        self, 
        kind: Literal['data', 'signals']='data',
        *,
        category: DataCategory | str=DataCategory.MARKET_DATA,
        pivot_data: bool=False,
        to_native: bool=True,
    ) -> NativeDataFrame | nw.DataFrame[Any]:
        """Returns one of the stored dataframes in either trading store or data stores.
    
        Args:
            kind: Which frame to return.
                - 'data': input dataframe from a data store (e.g. market data, news).
                - 'signals': signals produced by this component.
            category: For kind='data', which data category to return. 
                Ignored when kind='signals'. 
                Defaults to market data.
            pivot_data: pivot dataframe (when kind='data') to wide form. 
                Ignored when kind='signals'.
                Defaults to False.
            to_native: If True, return the underlying backend frame (polars/pandas) instead
                of a Narwhals DataFrame. Defaults to True.
        """
        if kind == 'data':
            category = DataCategory[category.upper()]
            data_store = self.get_data_store(category)
            df = data_store.df  # in long form
        elif kind == 'signals':
            df = self.trading_store.df  # in long/wide form, depends on the component's df_form setting
        else:
            raise ValueError(f'{kind=} is not supported')
        
        # pivot to wide form (only meaningful for data because data are stored in long form)
        # REVIEW: this requires dynamic pivoting for EACH call, not efficient
        if kind == 'data' and pivot_data:
            cols = df.columns
            if 'resolution' in cols and 'product' in cols:
                df = (
                    df
                    .pivot(
                        on=['resolution', 'product'],
                        index=['date'],
                    )
                    .sort('date')
                )
            else:
                # TODO: handle other data categories e.g. news data
                raise Exception(f'Unhandled data category {category}, cannot pivot to wide form')
        return df.to_native() if to_native else df
    
    def _materialize(self):
        data_dfs: dict[DataCategory, nw.DataFrame[Any]] = {}
        for data_category, data_store in self.data_stores.items():
            df: nw.DataFrame[Any] = data_store.materialize()
            data_dfs[data_category] = df
        signals_df: nw.DataFrame[Any] = self._component.signalize()
        self.trading_store.materialize(data_dfs, signals_df)
    
    def _flush_stale_bar(self, data: BarData):
        if data.is_closed():
            self._deliver(data)
    
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
        
        engine_name = self._component._context.name
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
        # set the ZMQPushHandler's receiver ready to flush the buffered log messages
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
    
    def _collect(self, msg: StreamingMessage | None=None):
        '''
        Args:
            msg: StreamingMessage, only provided when data is not being sent via zeromq
        '''
        # when not using zeromq (guaranteed ALL components are local components)
        if msg is not None:
            # TODO: update market_data_store from msg
            for component in self._component.get_components():
                component.databoy._collect(msg=msg)
            print('***databoy _collect:', msg)
        # when using zeromq (there could be some local and remote components, but both use zeromq to receive data anyways)
        else:
            from msgspec import structs
            while self._component.is_running():
                if msg_tuple := self._data_zmq.recv():
                    # TODO: how to know which data store to use from msg_tuple?
                    channel, topic, msg, msg_ts = msg_tuple
                    
                    # TEMP
                    print('databoy data_zmq recv:', channel, topic, msg, msg_ts)
                    
                    # TODO
                    # update = structs.asdict(msg)
                    # if topic == PublicDataChannel.orderbook:
                    #     self.market_data_store.update_quote(update)
                    # elif topic == PublicDataChannel.tradebook:
                    #     self.market_data_store.update_tick(update)
                    # elif topic == PublicDataChannel.candlestick:
                    #     self.market_data_store.update_bar(update)
                    # else:
                    #     raise NotImplementedError(f'{topic=} is not supported')
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
            elif data.resolution.is_tick():
                self._component._on_tick(data)
            elif data.resolution.is_bar():
                self._component._on_bar(data)
