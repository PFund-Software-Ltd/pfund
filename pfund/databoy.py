from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.messaging import ZeroMQ, LocalMQ
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.typing import ProductName, ResolutionRepr, DataConfigDict

import time
import importlib
from logging import Logger
from collections import defaultdict
from pprint import pformat

from pfund.products.product_base import BaseProduct
from pfund.datas.data_config import DataConfig
from pfund.datas.resolution import Resolution
from pfund.enums import Event, Broker, CryptoExchange, RunMode


class DataBoy:
    def __init__(self):
        self._logger: Logger | None = None
        self._datas: dict[BaseProduct, dict[Resolution, TimeBasedData]] = defaultdict(dict)
        self._listeners: dict[BaseData, list[BaseStrategy]] = defaultdict(list)
        self._stale_bar_timeouts: dict[BaseData, int] = {}
        self._queue: ZeroMQ | None = None
        # TODO: save data signatures properly, data_signatures should be a set
        # TODO: add data_config (dict form) to data_signatures
        # TODO: rename data_signatures to data_inputs?
        self._data_signatures = []
        
    def _set_logger(self, logger: Logger):
        self._logger = logger
    
    def _add_listener(self, listener: BaseStrategy, data: BaseData):
        if listener not in self._listeners[data]:
            self._listeners[data].append(listener)

    def _remove_listener(self, listener: BaseStrategy, data: BaseData):
        if listener in self._listeners[data]:
            self._listeners[data].remove(listener)
    
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
        from pfund.datas import QuoteData, TickData, BarData
        if resolution.is_quote():
            data = QuoteData(product, resolution, orderbook_depth=data_config.orderbook_depth, fast_orderbook=data_config.fast_orderbook)
        elif resolution.is_tick():
            data = TickData(product, resolution)
        else:
            data = BarData(product, resolution, shift=data_config.shift.get(resolution, 0), skip_first_bar=data_config.skip_first_bar.get(resolution, True))
            self._stale_bar_timeouts[data] = data_config.stale_bar_timeout[resolution]
        self._datas[product][resolution] = data
        return data
    
    def get_data(self, product: BaseProduct, resolution: ResolutionRepr | Resolution) -> TimeBasedData | None:
        if isinstance(resolution, str):
            resolution = Resolution(resolution)
        return self._datas[product].get(resolution, None)
    
    def remove_data(self, product: BaseProduct, resolution: ResolutionRepr | Resolution) -> TimeBasedData:
        if data := self.get_data(product, resolution):
            del self._datas[product][data.resolution]
            self._logger.debug(f'removed {product} {data.resolution} data')
            return data
        else:
            raise ValueError(f'{product} {resolution} data not found')

    def add_data(self, product: BaseProduct, data_config: DataConfigDict | DataConfig | None) -> list[TimeBasedData]:
        if not isinstance(data_config, DataConfig):
            data_config = DataConfig(**(data_config or {}))
        supported_resolutions = self._get_supported_resolutions(product.bkr, product.exch)
        is_auto_resampled = data_config.auto_resample(supported_resolutions)
        if is_auto_resampled:
            self._logger.warning(f'{product} resolution={data_config.primary_resolution} extra_resolutions={data_config.extra_resolutions} data is auto-resampled to:\n{pformat(data_config.resample)}')
        
        datas: list[TimeBasedData] = []
        for resolution in data_config.resolutions:
            if not (data := self.get_data(product, resolution)):
                data = self._add_data(product, resolution, data_config)
            datas.append(data)
        
        # mutually bind data_resampler and data_resamplee
        for resamplee_resolution, resampler_resolution in data_config.resample.items():
            data_resamplee = self.get_data(product, resamplee_resolution)
            data_resampler = self.get_data(product, resampler_resolution)
            data_resamplee.bind_resampler(data_resampler)
            self._logger.debug(f'{product} resolution={resampler_resolution} (resampler) added listener resolution={resamplee_resolution} (resamplee) data')
        return datas

    def update_quote(self, product: ProductName, quote: dict):
        ts = quote['ts']
        update = quote['data']
        extra_data = quote['extra_data']
        bids, asks = update['bids'], update['asks']
        data = self.get_data(product, '1q')
        data.on_quote(bids, asks, ts, **extra_data)
        self.push(data, event=Event.quote, **extra_data)

    def update_tick(self, product: ProductName, tick: dict):
        update = tick['data']
        extra_data = tick['extra_data']
        px, qty, ts = update['px'], update['qty'], update['ts']
        data = self.get_data(product, '1t')
        data.on_tick(px, qty, ts, **extra_data)
        self.push(data, event=Event.tick, **extra_data)

    def update_bar(self, product: ProductName, bar: dict, is_incremental: bool=True):
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
            self.push(data, event=Event.bar, **extra_data)
        else:
            if data.is_ready(now=ts):
                self.push(data, event=Event.bar, **extra_data)
            data.on_bar(o, h, l, c, v, ts, is_incremental=is_incremental, **extra_data)
    
    def _flush_stale_bar(self, data: BaseData):
        if data.is_ready():
            self.push(data, event=Event.bar)
    
    def schedule_jobs(self, scheduler: BackgroundScheduler):
        for data, timeout in self._stale_bar_timeouts.items():
            scheduler.add_job(lambda: self._flush_stale_bar(data), 'interval', seconds=timeout)
    
    def _setup_messaging(self, run_mode: RunMode):
        import zmq
        from pfund.messaging import ZeroMQ, LocalMQ
        if run_mode == RunMode.REMOTE:
            zmq_urls = self._engine.settings.zmq_urls
            self._queue = ZeroMQ(
                url=zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL),
                receiver_socket_type=zmq.SUB,  # receive data from engine
                sender_socket_type=zmq.PUSH,  # send e.g. orders to engine
            )
            # TODO: subscribe to selected topics, e.g. b'BYBIT:orderbook:BTCUSDT'
            self._queue.setsockopt(zmq.SUBSCRIBE, b'')
        else:
            self._queue = LocalMQ()
    
    # FIXME
    def pong(self):
        """Pongs back to Engine's ping to show that it is alive"""
        zmq_msg = (0, 0, (self.strat,))
        self._zmq.send(*zmq_msg, receiver='engine')

    # FIXME: to be removed
    def start_zmq(self):
        zmq_ports = self._engine.zmq_ports
        self._zmq = ZeroMQ(self.name)
        self._zmq.start(
            logger=self.logger,
            send_port=zmq_ports[self.name],
            recv_ports=[zmq_ports['engine']]
        )
        zmq_msg = (0, 1, (self.strat, os.getpid(),))
        self._zmq.send(*zmq_msg, receiver='engine')

    def stop_zmq(self):
        self._zmq.stop()
        self._zmq = None
        
    # TODO:
    def _collect(self):
        pass
    
    def _deliver(self, data: BaseData, event: Event, **extra_data):
        for strategy in self._listeners[data]:
            if not self._engine._use_ray:
                if strategy.is_running():
                    if event == Event.quote:
                        strategy.update_quote(data, **extra_data)
                        for data_resamplee in data.get_resamplees():
                            self.push(data_resamplee, event=event)
                    elif event == Event.tick:
                        strategy.update_tick(data, **extra_data)
                        for data_resamplee in data.get_resamplees():
                            self.push(data_resamplee, event=event)
                    elif event == Event.bar:
                        strategy.update_bar(data, **extra_data)
                        for data_resamplee in data.get_resamplees():
                            if data_resamplee.is_ready(now=data.end_ts) and not data_resamplee.skip_first_bar():
                                self.push(data_resamplee, event=event)
                        data.clear()
            else:
                raise NotImplementedError('parallel strategy is not implemented')
                # TODO
                # self._zmq
        
    # TODO: convert topic to enum
    # TODO: write data to duckdb files
    def handle_msgs(self, topic, msg: tuple):
        if topic == 1:  # quote data
            bkr, exch, pdt, quote = msg
            product = self._broker.get_product(exch=exch, pdt=pdt)
            self.update_quote(product, quote)
        elif topic == 2:  # tick data
            bkr, exch, pdt, tick = msg
            product = self._broker.get_product(exch=exch, pdt=pdt)
            self.update_tick(product, tick)
        elif topic == 3:  # bar data
            bkr, exch, pdt, bar = msg
            product = self._broker.get_product(exch=exch, pdt=pdt)
            self.update_bar(product, bar, now=time.time())