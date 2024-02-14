# NOTE: need this to make TYPE_CHECKING work to avoid the circular import issue
from __future__ import annotations

import time
import datetime
import copy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.feeds.base_feed import BaseFeed
    from pfund.datas.data_base import BaseData

from pfund.managers.data_manager import get_resolutions_from_kwargs
from pfund.models.model_backtest import BacktestModel
from pfund.models.model_base import BaseModel


_PFUND_BACKTEST_KWARGS = ['data_source', 'rollback_period', 'start_date', 'end_date']
_EVENT_DRIVEN_BACKTEST_KWARGS = ['resamples', 'shifts', 'auto_resample']


class BacktestMixin:
    def add_data(self, trading_venue, base_currency, quote_currency, ptype, *args, backtest: dict | None=None, train: dict | None=None, **kwargs) -> list[BaseData]:
        backtest_kwargs, train_kwargs = backtest or {}, train or {}
        
        if backtest_kwargs:
            data_source = self._get_data_source(trading_venue, backtest_kwargs)
            feed = self.get_feed(data_source)
            kwargs = self._prepare_kwargs(feed, kwargs)
        
        datas = super().add_data(trading_venue, base_currency, quote_currency, ptype, *args, **kwargs)
        
        if train_kwargs:
            self.set_data_periods(datas, **train_kwargs)

        if backtest_kwargs:
            dfs = self.get_historical_data(feed, datas, kwargs, copy.deepcopy(backtest_kwargs))
            for data, df in zip(datas, dfs):
                self.add_raw_df(data, df)
        return datas
        
    def add_model(self, model: BaseModel, name: str='', model_path: str='', is_load: bool=True) -> BaseModel:
        name = name or model.__class__.__name__
        model = BacktestModel(type(model), model.ml_model, *model._args, **model._kwargs)
        return super().add_model(model, name=name, model_path=model_path, is_load=is_load)
        
    def _get_data_source(self, trading_venue: str, backtest_kwargs: dict):
        from pfeed.const.commons import SUPPORTED_DATA_FEEDS
        trading_venue = trading_venue.upper()
        # if data_source is not defined, use trading_venue as data_source
        if trading_venue in SUPPORTED_DATA_FEEDS and 'data_source' not in backtest_kwargs:
            backtest_kwargs['data_source'] = trading_venue
        assert 'data_source' in backtest_kwargs, f"data_source must be defined"
        data_source = backtest_kwargs['data_source'].upper()
        assert data_source in SUPPORTED_DATA_FEEDS, f"{data_source=} not in {SUPPORTED_DATA_FEEDS}"
        return data_source
    
    def _prepare_kwargs(self, feed: BaseFeed, kwargs: dict):
        assert 'resolution' in kwargs or 'resolutions' in kwargs, f"data resolution(s) must be defined for {feed.name}"
        
        if self._Engine.mode == 'vectorized':
            # clear kwargs that are only for event driven backtesting
            for k in _EVENT_DRIVEN_BACKTEST_KWARGS:
                if k == 'auto_resample':
                    kwargs[k] = {'by_official_resolution': False, 'by_highest_resolution': False}
                else:
                    kwargs[k] = {}
        elif self._Engine.mode == 'event_driven':
            if 'is_skip_first_bar' not in kwargs:
                kwargs['is_skip_first_bar'] = False
        
            # add 'shifts' to kwargs:
            # HACK: since Yahoo Finance hourly data starts from 9:30 to 10:30 etc.
            # shift the start_ts (e.g. 9:00) of the bar to 30 minutes
            if feed.name == 'YAHOO_FINANCE':
                if 'shifts' not in kwargs:
                    kwargs['shifts'] = {}  # e.g. kwargs['shifts'] = {'1h': 30}
                for resolution in get_resolutions_from_kwargs(kwargs):
                    if resolution.is_hour() and repr(resolution) not in kwargs['shifts']:
                        # REVIEW: is there a better way to automatically determine the shifts? instead of hard-coding it to be 30 for yfinance here
                        kwargs['shifts'][repr(resolution)] = 30

        # override supported timeframes and periods using feed's
        # e.g. user might use IB as a broker for backtesting, but use Yahoo Finance as a data source
        # so IB's supported timeframes and periods should be overridden by Yahoo Finance's
        if hasattr(feed, 'SUPPORTED_TIMEFRAMES_AND_PERIODS'):
            kwargs['supported_timeframes_and_periods'] = feed.SUPPORTED_TIMEFRAMES_AND_PERIODS
        
        return kwargs
    
    @staticmethod
    def _remove_pfund_backtest_kwargs(kwargs: dict, backtest_kwargs: dict):
        '''backtest_kwargs include kwargs for both pfund backtesting and data feeds such as yfinance,
        clear pfund's kwargs for backtesting, only kwargs for e.g. yfinance are left
        '''
        for k in _PFUND_BACKTEST_KWARGS:
            assert k not in kwargs, f"kwarg '{k}' should be put inside 'backtest={{'{k}': '{kwargs[k]}'}}' "
            # clear PFund's kwargs for backtesting, only kwargs for e.g. yfinance are left
            if k in backtest_kwargs:
                del backtest_kwargs[k]
        return backtest_kwargs

    @staticmethod
    def get_feed(data_source: str) -> BaseFeed:
        from pfeed.feeds import YahooFinanceFeed, BybitFeed
        data_source = data_source.upper()
        if data_source == 'YAHOO_FINANCE':
            feed = YahooFinanceFeed()
        elif data_source == 'BYBIT':
            feed = BybitFeed()
        # TODO: other feeds
        else:
            raise NotImplementedError
        return feed

    def get_historical_data(self, feed: BaseFeed, datas: list[BaseData], kwargs: dict, backtest_kwargs: dict):
        rollback_period = backtest_kwargs.get('rollback_period', '1M')
        start_date = backtest_kwargs.get('start_date', '')
        end_date = backtest_kwargs.get('end_date', '')
        backtest_kwargs = self._remove_pfund_backtest_kwargs(kwargs, backtest_kwargs)
        
        dfs = []
        rate_limit = 3  # in seconds, 1 request every x seconds
        utcnow_date = str(datetime.datetime.now(tz=datetime.timezone.utc).date())
        for n, data in enumerate(datas):
            if data.is_time_based():
                if data.is_resamplee():
                    continue
            product = data.product
            resolution = data.resolution
            pdt_or_symbol = product.symbol if feed.name == 'YAHOO_FINANCE' else product.pdt
            df = feed.get_historical_data(pdt_or_symbol, rollback_period=rollback_period, start_date=start_date, end_date=end_date, resolution=repr(resolution), **backtest_kwargs)
            assert not df.empty, f"dataframe is empty for {pdt_or_symbol=}"
            df.reset_index(inplace=True)
            latest_date = str(df['ts'][0])
            if feed.name == 'YAHOO_FINANCE' and latest_date == utcnow_date:
                print(f'To avoid non-deterministic metrics, Yahoo Finance data on {latest_date} is removed since it might be being updated in real-time')
                df = df[:-1]
            
            df['product'] = product
            df['resolution'] = resolution
            dfs.append(df)
            
            # don't sleep on the last one loop, waste of time
            if feed.name == 'YAHOO_FINANCE' and n != len(datas) - 1:
                time.sleep(rate_limit)
        return dfs