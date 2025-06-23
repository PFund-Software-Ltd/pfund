from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from logging import Logger
    from mtflow.messaging.zeromq import ZeroMQ
    from pfeed.typing import tDataSource
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.typing import (
        ComponentName, 
        Component, 
        ProductName, 
        ResolutionRepr, 
        ZeroMQName, 
        DataConfigDict,
        EngineName, 
        tTradingVenue,
    )

import time
import importlib
from collections import defaultdict
from pprint import pformat

from pfund.datas import QuoteData, TickData, BarData
from pfund.products.product_base import BaseProduct
from pfund.datas.data_config import DataConfig
from pfund.datas.resolution import Resolution
from pfund.enums import Event, Broker, CryptoExchange, PFundDataChannel, InternalTopic


MarketData = QuoteData | TickData | BarData


# NOTE: conceptually it's similar to the messenger in engine.
class DataBoy:
    def __init__(self, component: Component):
        '''
        Args:
            datas: datas directly used by the component, added via add_data()
            _subscribed_data: all data in `datas` + data subscribed on behalf of local components
        '''
        self._component: Component = component
        self.datas: dict[BaseProduct, dict[Resolution, MarketData]] = defaultdict(dict)
        self._subscribed_data: dict[MarketData, list[Component]] = defaultdict(list)
        self._stale_bar_timeouts: dict[BarData, int] = {}
        self._data_zmq: ZeroMQ | None = None
        self._signals_zmq: ZeroMQ | None = None
        # TODO: save data signatures properly, data_signatures should be a set
        # TODO: add data_config (dict form) to data_signatures
        # TODO: rename data_signatures to data_inputs?
        self._data_signatures = []
    
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
    def logger(self) -> Logger:
        return self._component.logger
    
    def is_remote(self) -> bool:
        return self._component.is_remote()
        
    def _subscribe_to_data(self, component: Component, data: MarketData):
        self._subscribed_data[data].append(component)
        
    @staticmethod
    def _get_supported_resolutions(bkr: Broker, exch: CryptoExchange | str) -> dict:
        if bkr == Broker.CRYPTO:
            WebsocketApi = getattr(importlib.import_module(f'pfund.exchanges.{exch.lower()}.ws_api'), 'WebsocketApi')
            supported_resolutions = WebsocketApi.SUPPORTED_RESOLUTIONS
        elif bkr == Broker.IB:
            IBApi = getattr(importlib.import_module('pfund.brokers.ib.ib_api'), 'IBApi')
            supported_resolutions = IBApi.SUPPORTED_RESOLUTIONS
        else:
            raise NotImplementedError(f'{bkr=} is not supported')
        return supported_resolutions
    
    def _add_data(self, product: BaseProduct, resolution: Resolution, data_config: DataConfig) -> TimeBasedData:
        if resolution.is_quote():
            data = QuoteData(
                data_source=data_config.data_source, 
                data_origin=data_config.data_origin, 
                product=product, 
                resolution=resolution
            )
        elif resolution.is_tick():
            data = TickData(
                data_source=data_config.data_source, 
                data_origin=data_config.data_origin, 
                product=product, 
                resolution=resolution
            )
        else:
            data = BarData(
                data_source=data_config.data_source, 
                data_origin=data_config.data_origin, 
                product=product, 
                resolution=resolution, 
                shift=data_config.shift.get(resolution, 0), 
                skip_first_bar=data_config.skip_first_bar.get(resolution, True)
            )
            self._stale_bar_timeouts[data] = data_config.stale_bar_timeout[resolution]
        self.datas[product][resolution] = data
        return data
    
    def get_data(self, product: BaseProduct, resolution: ResolutionRepr | Resolution) -> MarketData | None:
        resolution = Resolution(resolution)
        return self.datas[product].get(resolution, None)
    
    def add_data(
        self, 
        product: BaseProduct, 
        data_source: tDataSource, 
        data_origin: str, 
        data_config: DataConfigDict | DataConfig | None
    ) -> list[MarketData]:
        if not isinstance(data_config, DataConfig):
            data_config = data_config or {}
            data_config['primary_resolution'] = self._component._resolution
            data_config['data_source'] = data_source
            data_config['data_origin'] = data_origin
            data_config = DataConfig(**data_config)
        supported_resolutions = self._get_supported_resolutions(product.bkr, product.exch)
        is_auto_resampled = data_config.auto_resample(supported_resolutions)
        if is_auto_resampled:
            self.logger.warning(f'{product} resolution={data_config.primary_resolution} extra_resolutions={data_config.extra_resolutions} data is auto-resampled to:\n{pformat(data_config.resample)}')
        
        datas: list[MarketData] = []
        for resolution in data_config.resolutions:
            if not (data := self.get_data(product, resolution)):
                data = self._add_data(product, resolution, data_config)
            datas.append(data)
        
        # mutually bind data_resampler and data_resamplee
        for resamplee_resolution, resampler_resolution in data_config.resample.items():
            data_resamplee = self.get_data(product, resamplee_resolution)
            data_resampler = self.get_data(product, resampler_resolution)
            data_resamplee.bind_resampler(data_resampler)
            self.logger.debug(f'{product} resolution={resampler_resolution} (resampler) added listener resolution={resamplee_resolution} (resamplee) data')
        
        return datas

    def _update_quote(self, product: ProductName, quote: dict):
        ts = quote['ts']
        update = quote['data']
        extra_data = quote['extra_data']
        bids, asks = update['bids'], update['asks']
        data = self.get_data(product, '1q')
        data.on_quote(bids, asks, ts, **extra_data)
        self._deliver(data, event=Event.quote, **extra_data)

    def _update_tick(self, product: ProductName, tick: dict):
        update = tick['data']
        extra_data = tick['extra_data']
        px, qty, ts = update['px'], update['qty'], update['ts']
        data = self.get_data(product, '1t')
        data.on_tick(px, qty, ts, **extra_data)
        self._deliver(data, event=Event.tick, **extra_data)

    def _update_bar(self, product: ProductName, bar: dict, is_incremental: bool=True):
        '''
        Args:
            is_incremental: if True, the bar update is incremental, otherwise it is a full bar update
                some exchanges may push incremental bar updates, some may only push when the bar is complete
        '''
        resolution: ResolutionRepr = bar['resolution']
        update = bar['data']
        extra_data = bar['extra_data']
        o, h, l, c, v, ts = update['open'], update['high'], update['low'], update['close'], update['volume'], update['ts']
        data = self.get_data(product, resolution)
        if not is_incremental:  # means the bar is complete
            data.on_bar(o, h, l, c, v, ts, is_incremental=is_incremental, **extra_data)
            self._deliver(data, event=Event.bar, **extra_data)
        else:
            if data.is_ready(now=ts):
                self._deliver(data, event=Event.bar, **extra_data)
            data.on_bar(o, h, l, c, v, ts, is_incremental=is_incremental, **extra_data)
    
    def _flush_stale_bar(self, data: BaseData):
        if data.is_ready():
            self._deliver(data, event=Event.bar)
    
    def schedule_jobs(self, scheduler: BackgroundScheduler):
        for data, timeout in self._stale_bar_timeouts.items():
            scheduler.add_job(lambda: self._flush_stale_bar(data), 'interval', seconds=timeout)
    
    def _setup_messaging(
        self, 
        zmq_urls: dict[EngineName | tTradingVenue | ComponentName, str], 
        zmq_ports: dict[ZeroMQName, int]
    ) -> dict[str, int]:
        '''
        Returns:
            zmq_ports_in_use: dict[str, int] of zmq ports in use by the component
        '''
        import zmq
        from mtflow.messaging import ZeroMQ
        
        zmq_url = zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL)
        
        data_zmq_name = self.name+'_data'
        self._data_zmq = ZeroMQ(
            name=data_zmq_name,
            url=zmq_url,
            port=zmq_ports.get(data_zmq_name, None),
            receiver_socket_type=zmq.SUB,  # receive data from engine
            sender_socket_type=zmq.PUSH,  # send component created data (e.g. orders) to engine
        )
        
        signals_zmq_name = self.name
        self._signals_zmq = ZeroMQ(
            name=signals_zmq_name,
            url=zmq_url,
            port=zmq_ports.get(signals_zmq_name, None),
            receiver_socket_type=zmq.SUB,  # subscribe to signals from other components
            sender_socket_type=zmq.PUB,  # publish signals to other consumers
        )
        
        zmq_ports_in_use = {q.name: q.port for q in [self._data_zmq, self._signals_zmq]}
        return zmq_ports_in_use
    
    def _setup_subscriptions(self, zmq_ports: dict[ZeroMQName, int]):
        # subscribe to engine's data proxied from trading venues (e.g. bybit's websocket)
        engine_proxy_port = zmq_ports['proxy']
        for data in self._subscribed_data:
            self._data_zmq.subscribe(engine_proxy_port, channel=data.zmq_channel)
            self.logger.debug(f'{self.name} subscribed to {data.zmq_channel} on port {engine_proxy_port}')

        for component in self.components:
            component_port = zmq_ports[component.name]
            self._signals_zmq.subscribe(component_port)
            self.logger.debug(f'{self.name} subscribed to {component.name} on port {component_port}')
    
    # FIXME
    def pong(self):
        """Pongs back to Engine's ping to show that it is alive"""
        zmq_msg = (0, 0, (self.strat,))
        self._zmq.send(*zmq_msg, receiver='engine')

    def start_zmq(self):
        if self._data_zmq:
            self._data_zmq.start()
        if self._signals_zmq:
            self._signals_zmq.start()

    def stop_zmq(self):
        self._zmq.stop()
        self._zmq = None
        
    def _collect(self, local_data=None):
        if self._data_zmq:
            channel, topic, data, pub_ts = self._data_zmq.recv()
            print(f'{self.name} recv:', channel, topic, data, pub_ts)
            # TODO: e.g. if component is a model:
            # output = self._component.predict(...)
            # self._signals_zmq.send(output)
        else:
            # TODO: listener.databoy._collect()
            if topic == 1:  # quote data
                bkr, exch, pdt, quote = msg
                product = self._component.get_product(exch=exch, pdt=pdt)
                data = self.get_data(product, '1q')
                self._component._on_quote(data)
            elif topic == 2:  # tick data
                bkr, exch, pdt, tick = msg
                product = self._component.get_product(exch=exch, pdt=pdt)
                data = self.get_data(product, '1t')
                self._component._on_tick(data)
            elif topic == 3:  # bar data
                bkr, exch, pdt, bar = msg
                product = self._component.get_product(exch=exch, pdt=pdt)
                data = self.get_data(product, ...)
                self._component._on_bar(data, now=time.time())
    
    def _deliver(self, data: BaseData, event: Event, **extra_data):
        # TODO
        if self._component.is_remote():
            raise NotImplementedError('parallel strategy is not implemented')
            # self._zmq
        else:
            if self._component.is_running():
                if event == Event.quote:
                    self._component._update_quote(data, **extra_data)
                    for data_resamplee in data.get_resamplees():
                        self._deliver(data_resamplee, event=event)
                elif event == Event.tick:
                    self._component._update_tick(data, **extra_data)
                    for data_resamplee in data.get_resamplees():
                        self._deliver(data_resamplee, event=event)
                elif event == Event.bar:
                    self._component._update_bar(data, **extra_data)
                    for data_resamplee in data.get_resamplees():
                        if data_resamplee.is_ready(now=data.end_ts) and not data_resamplee.skip_first_bar():
                            self._deliver(data_resamplee, event=event)
                    data.clear()
