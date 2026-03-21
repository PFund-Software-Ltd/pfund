# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from narwhals.typing import Frame, IntoFrame
    from pfeed.typing import GenericFrame

from pfeed.enums import DataLayer, DataAccessType, DataStorage
from pfeed.feeds.market_feed import MarketFeed
from pfund.enums import SourceType
from pfund.datas.data_market import MarketData
from pfund.datas.stores.base_data_store import BaseDataStore


class MarketDataStore(BaseDataStore[MarketData, MarketFeed]):
    # Columns pinned to the left side of the materialized dataframe for readability
    LEFT_COLS = ['date', 'resolution', 'product', 'symbol', 'source_type']
    
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

    def materialize(self) -> Frame:
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
        
        return nw.concat(dfs)
        
    # TODO:
    def swap_live_for_eod(self):
        '''Discard the interim live-stream buffer and load the official end-of-day dataset (if any).'''
        pass