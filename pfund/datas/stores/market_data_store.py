# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false
from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from narwhals.typing import Frame
    from pfeed.typing import GenericFrame

from pfeed.enums import DataLayer, DataAccessType, DataStorage
from pfeed.feeds.market_feed import MarketFeed
from pfund.enums import SourceType
from pfund.datas.data_market import MarketData
from pfund.datas.stores.base_data_store import BaseDataStore


class MarketDataStore(BaseDataStore[MarketData, MarketFeed]):
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

        dfs: list[GenericFrame] = []
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
                # pfund can only deal with cleaned data, must clean raw data in retrieval if raw data is stored
                clean_raw_data=True if storage_config.data_layer == DataLayer.RAW else False,
                **product_specs,
            )

            # check cache first for previously resampled data
            if settings.cache_materialized_data is not False:
                df: GenericFrame | None = (
                    retrieve(storage_config=cache_storage_config)
                    .run()
                )
                if df is not None:
                    self._logger.debug(f'Cache hit for {data.product.name} {data.resolution}')
                    dfs.append(df)
                    continue

            # cache miss or caching disabled, retrieve from original storage
            df: GenericFrame | None = (
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
            
            if df is None:
                if settings.auto_download_data:
                    # PAID data cannot be downloaded automatically, user must download it manually
                    if feed.data_source.access_type == DataAccessType.PAID_BY_USAGE:
                        raise DataNotFoundError(f'No data found for {data.product.name} {data.resolution}, and auto-downloading PAID data from {feed.data_source.name} is NOT allowed')

                    self._logger.warning(
                        f'No data found for {data.product.name} {data.resolution}, auto-downloading data...'
                    )
                    df: GenericFrame | None = (
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
                    if df is None:
                        raise DataNotFoundError(f'Failed to download data for {data.product.name} {data.resolution}')
                    else:
                        dfs.append(df)
                else:
                    raise DataNotFoundError(
                        f'No data found for {data.product.name} {data.resolution}.\n' +
                        "and 'auto_download_data' is disabled in settings, please enable it in engine settings or use 'pfeed' to download the data manually."
                    )
            else:
                dfs.append(df)
        
        df = nw.concat([nw.from_native(df) for df in dfs])
        df = df.with_columns(
            source_type=nw.lit(SourceType.BATCH).cast(nw.String)
        )
        return df
        
    # TODO:
    def swap_live_for_eod(self):
        '''Discard the interim live-stream buffer and load the official end-of-day dataset (if any).'''
        pass