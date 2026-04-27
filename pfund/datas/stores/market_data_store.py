# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportUnnecessaryComparison=false
from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, Any, TypeAlias
from functools import partial

if TYPE_CHECKING:
    from narwhals.typing import Frame, IntoFrame
    from pfeed.typing import GenericFrame
    from pfund.typing import ProductName
    from pfund.entities.products.product_base import BaseProduct
    from pfund.datas.data_config import DataConfig
    from pfund.datas.databoy import DataBoy
    ResolutionRepr: TypeAlias = str
    class BarUpdate(TypedDict, total=True):
        product: ProductName
        resolution: Resolution
        ts: float
        open: float
        high: float
        low: float
        close: float
        volume: float
        is_incremental: bool
        msg_ts: float | None
        extra_data: dict[str, Any]

import time
from collections import defaultdict

import narwhals as nw

from pfeed.enums import DataLayer, DataAccessType, DataStorage
from pfeed.storages.storage_config import StorageConfig
from pfeed.feeds.market_feed import MarketFeed
from pfund.datas.data_config import DataConfig
from pfund.datas.resolution import Resolution
from pfund.enums import SourceType, Environment
from pfund.datas.data_market import MarketData
from pfund.datas.stores.base_data_store import BaseDataStore
from pfund.datas import QuoteData, TickData, BarData


class MarketDataStore(BaseDataStore[MarketData, MarketFeed]):
    # Columns pinned to the left side of the materialized dataframe for readability
    LEFT_COLS = ['date', 'resolution', 'product', 'source_type']
    PIVOT_COLS = ['resolution', 'product']

    def __init__(self, databoy: DataBoy):
        super().__init__(databoy)
        self._datas: dict[ProductName, dict[ResolutionRepr, MarketData]] = defaultdict(dict)
        
    def get_data(self, product: ProductName, resolution: Resolution | ResolutionRepr) -> MarketData | None:
        if product not in self._datas:
            return None
        if isinstance(resolution, Resolution):
            resolution = repr(resolution)
        return self._datas[product].get(resolution, None)
    
    def get_datas(self) -> list[MarketData]:
        return list(set(
            data 
            for data_per_resolution in self._datas.values() 
            for data in data_per_resolution.values()
        ))
    
    def add_data(self, product: BaseProduct, data_config: DataConfig, storage_config: StorageConfig) -> list[MarketData]:
        datas: list[MarketData] = []
        for resolution in data_config.resolutions:
            data = self.get_data(product.name, resolution)
            if data is None:
                if storage_config.data_layer == DataLayer.RAW:
                    raise ValueError('Loading data from RAW data layer is not supported, pfund can only deal with cleaned data')
                data = self.create_data(product=product, resolution=resolution, data_config=data_config, storage_config=storage_config)
                self._datas[product.name][repr(resolution)] = data
            datas.append(data)
        
        # mutually bind data_resampler and data_resamplee
        for resamplee_resolution, resampler_resolution in data_config.resample.items():
            data_resamplee = self.get_data(product.name, resamplee_resolution)
            data_resampler = self.get_data(product.name, resampler_resolution)
            data_resamplee.bind_resampler(data_resampler)
            self._logger.debug(f'{product.name} resolution={resampler_resolution} (resampler) added listener resolution={resamplee_resolution} (resamplee) data')
        
        return datas
    
    def create_data(self, product: BaseProduct, resolution: Resolution, data_config: DataConfig, storage_config: StorageConfig) -> MarketData:
        if resolution.is_quote():
            data = QuoteData(
                data_source=data_config.data_source, 
                data_origin=data_config.data_origin, 
                product=product, 
                resolution=resolution,
                data_config=data_config,
                storage_config=storage_config
            )
        elif resolution.is_tick():
            data = TickData(
                data_source=data_config.data_source, 
                data_origin=data_config.data_origin, 
                product=product, 
                resolution=resolution,
                data_config=data_config,
                storage_config=storage_config
            )
        else:
            data = BarData(
                data_source=data_config.data_source, 
                data_origin=data_config.data_origin, 
                product=product, 
                resolution=resolution, 
                data_config=data_config,
                storage_config=storage_config,
            )
        return data
    
    # TODO
    def update_quote(self, update: QuoteUpdate):
        data = self.get_data(update['product'], update['resolution'])
        if data is None:
            return
        data.on_quote(update['bids'], update['asks'], update['ts'], **update['extra_data'])
        self._databoy._deliver(data)

    # TODO
    def update_tick(self, update: TickUpdate):
        data = self.get_data(update['product'], update['resolution'])
        if data is None:
            return
        data.on_tick(
            price=update['price'], volume=update['volume'], 
            ts=update['ts'], msg_ts=update['msg_ts'], 
            extra_data=update['extra_data']
        )
        self._databoy._deliver(data)
        if data_resamplees := data.get_resamplees():
            price = update.pop('price')
            volume = update.pop('volume')
            bar_update = update.copy()
            bar_update['is_incremental'] = True
            bar_update['open'] = price
            bar_update['high'] = price
            bar_update['low'] = price
            bar_update['close'] = price
            bar_update['volume'] = volume
            for data_resamplee in data_resamplees:
                bar_update['resolution'] = data_resamplee.resolution
                self.update_bar(bar_update)
        
    def update_bar(self, update: BarUpdate):
        '''update bar data from streaming message
        if ready, deliver the bar data to databoy
        '''
        def _update_data(data: MarketData):
            data.on_bar(
                o=update['open'], h=update['high'], l=update['low'], c=update['close'], v=update['volume'], ts=update['ts'], 
                msg_ts=update['msg_ts'], 
                extra_data=update['extra_data'],
                is_incremental=update['is_incremental']
            )

        data = self.get_data(update['product'], update['resolution'])
        # NOTE: when not using zeromq, ALL msgs are passed to all components. 
        # If this component didn't subscribe to this data (via add_data()), get_data() returns None and the update is skipped.
        if data is None:
            return 
        if not update['is_incremental']:
            _update_data(data)
            self._databoy._deliver(data)
            # NOTE: in case update['ts'] < data.end_ts but the bar is already closed
            # pick the max value so that resamplees can use the correct ts to determine if they are closed
            # e.g. data is 1m bar, resamplee is 15m bar
            # without this, resamplee might not know the bar is closed when update['ts'] < data.end_ts
            update['ts'] = max(update['ts'], data.end_ts)
        else:
            # deliver the closed bar before update() clears it for the next bar
            if not data.is_closed() and data.is_closed(now=update['ts'] or update['msg_ts'] or time.time()):
                self._databoy._deliver(data)
                _update_data(data)
            else:
                _update_data(data)
                if data.config.push_incomplete_bar:
                    self._databoy._deliver(data)

        # update resamplees
        if data_resamplees := data.get_resamplees():
            resamplee_update = update.copy()
            resamplee_update['is_incremental'] = True
            for data_resamplee in data_resamplees:
                resamplee_update['resolution'] = data_resamplee.resolution
                self.update_bar(resamplee_update)
    
    # TODO
    def update_df(self, data: MarketData):
        print('***update_df', data)
        print('***update_df', self._df)
    
    def _should_cache_resampled(self, feed: MarketFeed) -> bool:
        '''Determine if the retrieved data should be cached to the CURATED layer.'''
        engine_settings = self._databoy._component.context.settings
        setting = engine_settings.cache_materialized_data
        if setting is True:
            return True
        if setting is False:
            return False
        # 'auto': cache only when resampling actually occurred
        request = feed._current_request
        return request.data_resolution != request.target_resolution

    def materialize(self) -> nw.DataFrame[Any]:
        '''Materializes market data by loading from storage, with optional auto-download fallback.

        For each registered data feed, first checks the cache for previously resampled data.
        If not cached, retrieves from pfeed's data lakehouse, optionally caching the result.
        If data is not found and `auto_download_data` is enabled in settings, downloads it from source.
        Missing dates in partially available data are reported as warnings but not auto-filled.

        Raises:
            DataNotFoundError: If no data is found and auto-download is disabled,
                or if the data source is paid-by-usage (auto-download of paid data is not allowed).
        '''
        from pfeed.errors import DataNotFoundError
        
        def _prepare_df_before_append(data: MarketData, df: IntoFrame) -> Frame:
            '''Adds a 'product' column to the dataframe using product name'''
            nwdf = (
                nw
                .from_native(df)
                .with_columns(
                    product=nw.lit(data.product.name).cast(nw.String),
                    source_type=nw.lit(SourceType.BATCH).cast(nw.String)
                )
            )
            cols = nwdf.collect_schema().names()
            # re-order columns
            target_cols = self.LEFT_COLS + [col for col in cols if col not in self.LEFT_COLS]
            nwdf = nwdf.select(target_cols)
            return nwdf

        dfs: list[Frame] = []
        engine_context = self._databoy._component.context
        settings = engine_context.settings
        start_date, end_date = engine_context.data_start, engine_context.data_end

        for data in self.get_datas():
            # pfund only supports bar data as the main resolution
            if not data.is_bar():
                continue
            feed = self._create_feed(data)
            self._logger.debug(f'Materializing market data {data.product.name} {data.resolution}...')
            storage_config = data.storage_config
            product, symbol = str(data.product.basis), data.product.symbol
            product_specs = data.product.specs
            cache_storage_config = self._create_cache_storage_config(storage_config)
            retrieve = partial(
                feed.retrieve,
                env=Environment.BACKTEST,
                product=product,
                resolution=data.resolution,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_origin=data.origin,
                dataflow_per_date=None,  # setting it to None, pfeed will automatically determine it
                **product_specs,
            )

            # check cache first for previously resampled data
            if settings.cache_materialized_data is not False:
                _df: GenericFrame | None = (
                    retrieve(storage_config=cache_storage_config)
                    .run()
                )
                if _df is not None:
                    self._logger.info(f'loaded data from {storage_config.data_path} for {data.product.name} {data.resolution}')
                    dfs.append(_prepare_df_before_append(data, _df))
                    continue

            # cache miss or caching disabled, retrieve from original storage
            _df: GenericFrame | None = (
                retrieve(storage_config=storage_config)
                .load(
                    storage=DataStorage.CACHE if self._should_cache_resampled(feed) else None,
                    data_path=storage_config.data_path,
                    data_layer=DataLayer.CURATED,
                    io_format=storage_config.io_format,
                    compression=storage_config.compression,
                )
                .run()
            )
            if _df is not None:
                dfs.append(_prepare_df_before_append(data, _df))
                self._logger.info(f'loaded data from {storage_config.data_path} for {data.product.name} {data.resolution}')
            else:
                if settings.auto_download_data:
                    # PAID data cannot be downloaded automatically, user must download it manually
                    if feed.data_source.access_type == DataAccessType.PAID_BY_USAGE:
                        raise DataNotFoundError(f'No data found for {data.product.name} {data.resolution}, and auto-downloading PAID data from {feed.data_source.name} is NOT allowed')

                    self._logger.warning(
                        f'No data found for {data.product.name} {data.resolution}, auto-downloading data...'
                    )
                    _df: GenericFrame | None = (
                        feed
                        .download(
                            product=product,
                            resolution=data.resolution,
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            data_origin=data.origin,
                            storage_config=storage_config,
                            **product_specs
                        )
                        .run()
                    )
                    if _df is None:
                        raise DataNotFoundError(f'Failed to download data for {data.product.name} {data.resolution}')
                    else:
                        dfs.append(_prepare_df_before_append(data, _df))
                else:
                    raise DataNotFoundError(
                        f'No data found for {data.product.name} {data.resolution}.\n' +
                        "and 'auto_download_data' is disabled in settings, please enable it in engine settings or use 'pfeed' to download the data manually."
                    )

        df: Frame = nw.concat(dfs)
        if isinstance(df, nw.LazyFrame):
            df = df.collect()
        self._df = df
        cols = df.columns
        assert self.INDEX_COL in cols, f"Index column {self.INDEX_COL} not found in {cols}"
        assert all(col in cols for col in self.PIVOT_COLS), f"Pivot columns {self.PIVOT_COLS} not found in {cols}"
        return df
