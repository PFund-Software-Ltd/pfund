from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.datas.data_base import BaseData

import time
from collections import defaultdict
import importlib

from pfund.datas.resolution import Resolution
from pfund.datas import QuoteData, TickData, BarData
from pfund.products.product_base import BaseProduct
from pfund.managers.base_manager import BaseManager
        

def get_resolutions_from_kwargs(kwargs: dict) -> list[Resolution]:
    # create data based on resolution(s)
    key = 'resolutions' if 'resolutions' in kwargs else 'resolution'
    if type(kwargs[key]) is list:
        resolutions = kwargs[key]
    elif type(kwargs[key]) is str:
        resolutions = [kwargs[key]]
    else:
        raise Exception(f'{key} must be a list or str')
    resolutions = [Resolution(resolution) for resolution in set(resolutions)]  # use set() to remove duplicates
    return resolutions


class DataManager(BaseManager):
    def __init__(self, broker):
        super().__init__('data_manager', broker)
        # datas = {repr(product): {repr(resolution): data}}
        self._datas = defaultdict(dict)

    def _resample_to_official_resolution(self, product, resolution: Resolution, supported_timeframes_and_periods: dict | None=None) -> Resolution:
        """Resamples the resolution into an officially supported one
        e.g. 
            if '4m' is the resolution but only '1m' is supported officially,
            '4m' (input) will be resampled into '1m' (output)
        """
        if product.is_crypto():
            WebsocketApi = getattr(importlib.import_module(f'pfund.exchanges.{product.exch.lower()}.ws_api'), 'WebsocketApi')
            supported_timeframes_and_periods = supported_timeframes_and_periods or WebsocketApi.SUPPORTED_TIMEFRAMES_AND_PERIODS
        elif product.bkr == 'IB':
            IBApi = getattr(importlib.import_module('pfund.brokers.ib.ib_api'), 'IBApi')
            supported_timeframes_and_periods = supported_timeframes_and_periods or IBApi.SUPPORTED_TIMEFRAMES_AND_PERIODS
        # EXTEND
        else:
            pass
        period, timeframe = resolution.period, repr(resolution.timeframe)
        if (timeframe not in supported_timeframes_and_periods) or \
            (period == 1 and period not in supported_timeframes_and_periods[timeframe]):
            # change timeframe unit but retain the time value
            if resolution.is_second():  # REVIEW
                resolution_resampled = '1t'
            elif resolution.is_minute():
                resolution_resampled = '60s'
            elif resolution.is_hour():
                resolution_resampled = '60m'
            elif resolution.is_day():
                resolution_resampled = '24h'
            elif resolution.is_week():
                resolution_resampled = '7d'
            elif resolution.is_month():
                resolution_resampled = '4w'
            else:
                raise Exception(f'{resolution=} is not supported')
            resolution_resampled = Resolution(resolution_resampled)
            return self._resample_to_official_resolution(product, resolution_resampled, supported_timeframes_and_periods=supported_timeframes_and_periods)
        elif period not in supported_timeframes_and_periods[timeframe]:
            # resample by unit, e.g. '4m' is resampled by '1m'
            resolution_resampled = Resolution('1' + timeframe)
            return self._resample_to_official_resolution(product, resolution_resampled, supported_timeframes_and_periods=supported_timeframes_and_periods)
        else:
            return resolution
        
    def push(self, data, event, **kwargs):
        for listener in self._listeners[data]:
            strategy = listener
            if not strategy.is_parallel():
                if strategy.is_running():
                    if event == 'quote':
                        strategy.update_quote(data, **kwargs)
                    elif event == 'tick':
                        strategy.update_tick(data, **kwargs)
                    elif event == 'bar':
                        # for resampled data, the first bar is very likely incomplete
                        # so users can choose to skip it
                        if not data.is_skip_first_bar():
                            strategy.update_bar(data, **kwargs)
                        data.clear()
            # TODO
            # else:
            #     self._zmq

    def _create_time_based_data(self, product: BaseProduct, resolution: Resolution, **kwargs):
        if resolution.is_quote():
            data = QuoteData(product, resolution, **kwargs)
        elif resolution.is_tick():
            data = TickData(product, resolution, **kwargs)
        else:
            data = BarData(product, resolution, **kwargs)
        return data
    
    def _auto_resample(self, product: BaseProduct, resamples: dict[Resolution, Resolution], resolutions: list[Resolution], auto_resample: dict[str, bool], supported_timeframes_and_periods):
        def _auto_resample_by_highest_resolution():
            '''Resamples the resolutions automatically by using the highest resolution
            '''
            ascending_resolutions = sorted(resolutions, reverse=False)  # e.g. ['1d', '1h', '1m']
            highest_resolution = ascending_resolutions[-1]
            for resolution in ascending_resolutions[:-1]:
                if resolution not in resamples:
                    resamples[resolution] = highest_resolution
                    self.logger.debug(f'{product} {resolution} data is auto-resampled by {highest_resolution=}')
            return resamples
        
        def _auto_resample_by_official_resolution(resampler_resolutions: list[Resolution]) -> dict[Resolution, Resolution]:
            """Resamples the resolutions automatically if not supported officially.
            e.g. 
                if resolution is '4m', but only '1m' is supported officially,
                {'4m': '1m'} will be automatically added to resamples,
                meaning the '4m' bar will be created by '1m' bar.
            """
            auto_resamples = {}
            # check if all resolutions in datas are supported, if not, auto-resample
            for resolution in resampler_resolutions:
                # no resampling when timeframe is quote/tick
                if resolution.is_quote() or resolution.is_tick():
                    continue
                official_resolution = self._resample_to_official_resolution(product, resolution, supported_timeframes_and_periods=supported_timeframes_and_periods)
                # resolution has no change after resampling
                if resolution == official_resolution:
                    continue
                auto_resamples[resolution] = official_resolution
                self.logger.debug(f'{product} {resolution} data is auto-resampled by {official_resolution=}')
            return auto_resamples
        
        if auto_resample.get('by_highest_resolution', True):
            resamples = _auto_resample_by_highest_resolution()
            
        if auto_resample.get('by_official_resolution', True):
            resampler_resolutions = [resolution for resolution in resolutions if resolution not in resamples.keys()]
            auto_resamples = _auto_resample_by_official_resolution(resampler_resolutions)
            for resamplee_resolution, resampler_resolution in resamples.items():
                # resampler_resolution in auto_resamples = it is a resamplee_resolution resampeld by an official resolution
                if resampler_resolution in auto_resamples:
                    resampler_resolution_corrected_by_official_resolution = auto_resamples[resampler_resolution]
                    resamples[resamplee_resolution] = resampler_resolution_corrected_by_official_resolution
            resamples.update(auto_resamples)
        
        if resamples:
            self.logger.warning(f'{resamples=}')
        return resamples

    # TODO
    def add_custom_data(self):
        pass
    
    def set_data(self, product: BaseProduct, resolution: Resolution, data: BaseData):
        self._datas[repr(product)][repr(resolution)] = data

    def get_data(self, product: str | BaseProduct, resolution: str | Resolution) -> BaseData | None:
        if isinstance(product, BaseProduct):
            product = repr(product)
        if isinstance(resolution, Resolution):
            resolution = repr(resolution)
        return self._datas[product].get(resolution, None)
    
    def remove_data(self, product: BaseProduct, resolution: str) -> BaseData:
        if data := self.get_data(product, resolution):
            del self._datas[repr(product)][resolution]
            self.logger.debug(f'removed {product} {resolution} data')
            return data
        else:
            raise Exception(f'{product} {resolution} data not found')

    def add_data(self, product: BaseProduct, **kwargs) -> list[BaseData]:
        datas = []
        # time-based data
        if 'resolution' in kwargs or 'resolutions' in kwargs:
            resolutions: list[Resolution] = get_resolutions_from_kwargs(kwargs)
            if 'resolution' in kwargs:
                del kwargs['resolution']
            for resolution in resolutions:
                if not (data := self.get_data(product, resolution=resolution)):
                    data = self._create_time_based_data(product, resolution, **kwargs)
                    self.set_data(product, resolution, data)
                    datas.append(data)
            
            resamples = {Resolution(resamplee_resolution): Resolution(resampler_resolution) for resamplee_resolution, resampler_resolution in kwargs.get('resamples', {}).items()}
            default_auto_resample = {'by_official_resolution': True, 'by_highest_resolution': True}
            auto_resample = kwargs.get('auto_resample', default_auto_resample)
            supported_timeframes_and_periods = kwargs.get('supported_timeframes_and_periods', None)
            resamples = self._auto_resample(product, resamples, resolutions, auto_resample, supported_timeframes_and_periods)
                
            # mutually bind data_resampler and data_resamplee
            for resamplee_resolution, resampler_resolution in resamples.items():
                assert resamplee_resolution in resolutions, f'Your target resolution {resamplee_resolution=} must be included in kwarg {resolutions=}'
                if resampler_resolution <= resamplee_resolution:
                    raise Exception(f'Cannot use lower/equal resolution "{resampler_resolution}" to resample "{resamplee_resolution}"')
                if not (data_resampler := self.get_data(product, resolution=resampler_resolution)):
                    data_resampler = self._create_time_based_data(product, resampler_resolution)
                self.set_data(product, resampler_resolution, data_resampler)
                datas.append(data_resampler)
                self.logger.debug(f'added {product} data')
                data_resamplee = self.get_data(product, resolution=resamplee_resolution)
                data_resamplee.add_resampler(data_resampler)
                data_resampler.add_resamplee(data_resamplee)
                self.logger.debug(f'{product} {resampler_resolution} data added listener {resamplee_resolution} data')
        # TODO support volume-based etc.
        # elif True:
        #     pass
        else:
            raise Exception(f'{product} data resolution(s) must be defined')
        
        # FIXME: DEPRECATED, to be removed
        # datas: list[BaseData] = list(self._datas[repr(product)].values())
        return datas

    def update_quote(self, product: BaseProduct | str, quote: dict):
        ts = quote['ts']
        update = quote['data']
        other_info = quote['other_info']
        bids, asks = update['bids'], update['asks']
        data = self.get_data(product, resolution='1q')
        data.on_quote(bids, asks, ts, **other_info)
        self.push(data, event='quote', **other_info)
        for data_resamplee in data.get_resamplees():
            data_resamplee.on_quote(bids, asks, ts, **other_info)
            self.push(data_resamplee, event='quote', **other_info)

    def update_tick(self, product: BaseProduct | str, tick: dict):
        update = tick['data']
        other_info = tick['other_info']
        px, qty, ts = update['px'], update['qty'], update['ts']
        data = self.get_data(product, resolution='1t')
        data.on_tick(px, qty, ts, **other_info)
        self.push(data, event='tick', **other_info)
        for data_resamplee in data.get_resamplees():
            data_resamplee.on_tick(px, qty, ts, **other_info)
            self.push(data_resamplee, event='tick', **other_info)

    def update_bar(self, product: BaseProduct | str, bar: dict, now: int):
        resolution: str = bar['resolution']
        update = bar['data']
        other_info = bar['other_info']
        o, h, l, c, v, ts = update['open'], update['high'], update['low'], update['close'], update['volume'], update['ts']
        data = self.get_data(product, resolution=resolution)
        if data.is_ready(now=now):
            self.push(data, event='bar', **other_info)
        data.on_bar(o, h, l, c, v, ts, **other_info)
        for data_resamplee in data.get_resamplees():
            if data_resamplee.is_ready(now=now):
                self.push(data_resamplee, event='bar', **other_info)
            data_resamplee.on_bar(o, h, l, c, v, ts, is_volume_aggregated=True, **other_info)
            
    def handle_msgs(self, topic, info):
        if topic == 1:  # quote data
            bkr, exch, pdt, quote = info
            product = self._broker.get_product(exch=exch, pdt=pdt)
            self.update_quote(product, quote)
        elif topic == 2:  # tick data
            bkr, exch, pdt, tick = info
            product = self._broker.get_product(exch=exch, pdt=pdt)
            self.update_tick(product, tick)
        elif topic == 3:  # bar data
            bkr, exch, pdt, bar = info
            product = self._broker.get_product(exch=exch, pdt=pdt)
            self.update_bar(product, bar, now=time.time())