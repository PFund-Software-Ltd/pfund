# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, Any
from functools import partial

if TYPE_CHECKING:
    from narwhals.typing import Frame, IntoFrame
    from pfeed.typing import GenericFrame
    from pfund.typing import ProductName
    from pfund.entities.products.product_base import BaseProduct
    from pfund.engines.engine_context import EngineContext
    from pfund.datas.data_config import DataConfig
    class BarUpdate(TypedDict, total=True):
        ts: float
        open: float
        high: float
        low: float
        close: float
        volume: float
        is_incremental: bool
        msg_ts: float | None
        extra_data: dict[str, Any]

from collections import defaultdict

from pfeed.enums import DataLayer, DataAccessType, DataStorage
from pfeed.storages.storage_config import StorageConfig
from pfeed.feeds.market_feed import MarketFeed
from pfund.datas.data_config import DataConfig
from pfund.datas.resolution import Resolution
from pfund.enums import SourceType
from pfund.datas.data_market import MarketData
from pfund.datas.stores.base_data_store import BaseDataStore
from pfund.datas import QuoteData, TickData, BarData


class MarketDataStore(BaseDataStore[MarketData, MarketFeed]):
    # Columns pinned to the left side of the materialized dataframe for readability
    LEFT_COLS = ['date', 'resolution', 'product', 'symbol', 'source_type']

    def __init__(self, context: EngineContext):
        super().__init__(context)
        self._datas: dict[ProductName, dict[Resolution, MarketData]] = defaultdict(dict)
        self.stale_bar_timeouts: dict[BarData, int] = {}
    
    def get_data(self, product: ProductName, resolution: Resolution) -> MarketData | None:
        if product not in self._datas:
            return None
        return self._datas[product].get(resolution, None)
    
    def add_data(self, product: BaseProduct, storage_config: StorageConfig, data_config: DataConfig) -> list[MarketData]:
        datas: list[MarketData] = []
        for resolution in data_config.resolutions:
            data = self.get_data(product.name, resolution)
            if data is None:
                data = self.create_data(product, resolution, data_config)
                self._datas[product.name][resolution] = data
                if storage_config.data_layer == DataLayer.RAW:
                    raise ValueError('Loading data from RAW data layer is not supported, pfund can only deal with cleaned data')
                self._storage_configs[data] = storage_config
                if data.is_bar():
                    self.stale_bar_timeouts[data] = data_config.stale_bar_timeout[resolution]
                    # self._feeds is used in materialization, and pfund only supports bar data as the main resolution
                    self._feeds[data] = self._create_feed(data)
            datas.append(data)
        
        # mutually bind data_resampler and data_resamplee
        for resamplee_resolution, resampler_resolution in data_config.resample.items():
            data_resamplee = self.get_data(product.name, resamplee_resolution)
            data_resampler = self.get_data(product.name, resampler_resolution)
            data_resamplee.bind_resampler(data_resampler)
            self._logger.debug(f'{product.name} resolution={resampler_resolution} (resampler) added listener resolution={resamplee_resolution} (resamplee) data')
        
        return datas
    
    def create_data(self, product: BaseProduct, resolution: Resolution, data_config: DataConfig) -> MarketData:
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
        return data
    
    # TODO
    def update_quote(self, data: QuoteData, update: QuoteUpdate):
        ts = update['ts']
        extra_data = update['extra_data']
        bids, asks = update['bids'], update['asks']
        data.on_quote(bids, asks, ts, **extra_data)

    # TODO
    def update_tick(self, data: TickData, update: TickUpdate):
        extra_data = update['extra_data']
        px, qty, ts = update['px'], update['qty'], update['ts']
        data.on_tick(px, qty, ts, **extra_data)
    
    # FIXME: should get_data() in update_bar, pass in product and resolution to update_bar() instead of data
    def update_bar(self, data: BarData, update: BarUpdate):
        '''update bar data from streaming message
        if ready, deliver the bar data to the component
        '''
        data.on_bar(
            o=update['open'], h=update['high'], l=update['low'], c=update['close'], v=update['volume'], ts=update['ts'], 
            msg_ts=update['msg_ts'], 
            extra_data=update['extra_data'],
            is_incremental=update['is_incremental']
        )
    
    def _should_cache_resampled(self, feed: MarketFeed) -> bool:
        '''Determine if the retrieved data should be cached to the CURATED layer.'''
        setting = self._context.settings.cache_materialized_data
        if setting is True:
            return True
        if setting is False:
            return False
        # 'auto': cache only when resampling actually occurred
        request = feed._current_request
        return request.data_resolution != request.target_resolution

    def materialize(self) -> None:
        '''Materializes market data by loading from storage, with optional auto-download fallback.

        For each registered data feed, first checks the cache for previously resampled data.
        If not cached, retrieves from pfeed's data lakehouse, optionally caching the result.
        If data is not found and `auto_download_data` is enabled in settings, downloads it from source.
        Missing dates in partially available data are reported as warnings but not auto-filled.

        Raises:
            DataNotFoundError: If no data is found and auto-download is disabled,
                or if the data source is paid-by-usage (auto-download of paid data is not allowed).
        '''
        import narwhals as nw
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
            # re-order columns
            cols = nwdf.collect_schema().names()
            target_cols = self.LEFT_COLS + [col for col in cols if col not in self.LEFT_COLS]
            nwdf = nwdf.select(target_cols)
            return nwdf

        dfs: list[Frame] = []
        settings = self._context.settings
        start_date, end_date = self._context.data_start, self._context.data_end

        for data, feed in self._feeds.items():
            self._logger.debug(f'Materializing market data {data.product.name} {data.resolution}...')
            storage_config = self._storage_configs[data]
            product, symbol = str(data.product.basis), data.product.symbol
            product_specs = data.product.specs
            cache_storage_config = self._create_cache_storage_config(storage_config)
            retrieve = partial(
                feed.retrieve,
                env=self._context.env,
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
        
        self.df = nw.concat(dfs)
        
    # TODO:
    def swap_live_for_eod(self):
        '''Discard the interim live-stream buffer and load the official end-of-day dataset (if any).'''
        pass