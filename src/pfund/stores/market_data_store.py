from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from pfeed.dataflow.result import RunResult
    from pfund_blocks.datas import Bar, Quote, Tick
    from pfund_blocks.entities.products.product_base import BaseProduct
    from pfund_blocks.resolution import Resolution
    from pfund_blocks.typing import ProductName, ResolutionRepr

    from pfund.databoy import DataBoy

import time
from collections import defaultdict
from functools import partial

import narwhals as nw
from pfeed.enums import DataAccessType
from pfeed.market.feed import MarketFeed
from pfund_blocks.datas import BarData, MarketData, QuoteData, TickData
from pfund_blocks.enums.env import Environment
from pfund_blocks.resolution import Resolution

from pfund.components.bar_dataframe import (
    KEY_COLS,
    reorder_key_cols,
    validate_spine_df,
)
from pfund.data_config import DataConfig
from pfund.stores.base_data_store import BaseDataStore


class MarketDataStore(BaseDataStore[MarketData, MarketFeed]):
    METADATA_COLS: ClassVar[list[str]] = ["source_type"]

    def __init__(self, databoy: DataBoy):
        super().__init__(databoy)
        self._datas: dict[ProductName, dict[ResolutionRepr, MarketData]] = defaultdict(
            dict
        )

    def get_data(
        self, product: ProductName, resolution: Resolution | ResolutionRepr
    ) -> MarketData | None:
        if product not in self._datas:
            return None
        if isinstance(resolution, Resolution):
            resolution = repr(resolution)
        return self._datas[product].get(resolution, None)

    def get_datas(self) -> list[MarketData]:
        # sort to make the output deterministic
        return sorted(
            set(
                data
                for data_per_resolution in self._datas.values()
                for data in data_per_resolution.values()
            ),
            key=lambda data: (data.product.name, repr(data.resolution)),
        )

    def _add_data(
        self,
        product: BaseProduct,
        resolution: Resolution,
        config: DataConfig,
    ) -> MarketData:
        data = self.get_data(product.name, resolution)
        if data is not None:
            return data
        data = self.create_data(
            product=product,
            resolution=resolution,
            config=config,
        )
        self._datas[product.name][repr(resolution)] = data
        return data

    @staticmethod
    def _validate_resolutions(resolutions: list[Resolution]) -> None:
        """Raises if the resolutions together are not supported, e.g. more than one tick resolution."""
        quote_resolution_exists = False
        tick_resolution_exists = False
        for resolution in resolutions:
            if resolution.is_quote():
                if quote_resolution_exists:
                    raise ValueError("only one quote resolution is supported")
                quote_resolution_exists = True
            elif resolution.is_tick():
                if tick_resolution_exists:
                    raise ValueError("only one tick resolution is supported")
                tick_resolution_exists = True
            # REVIEW: support week, month, year?
            elif resolution.is_week() or resolution.is_month() or resolution.is_year():
                raise ValueError(
                    f"{resolution=} is not supported, resolution is at least daily"
                )

    def add_data(
        self,
        product: BaseProduct,
        resolutions: list[Resolution | str] | None = None,
        config: DataConfig | None = None,
        as_features: bool = False,
    ) -> list[MarketData]:
        self._set_data_as_features(as_features)
        config = config or DataConfig()
        data_resolutions: list[Resolution] = list(
            dict.fromkeys([
                self._component.resolution,
                *(Resolution(r) for r in resolutions or []),
                *config.get_resolutions(),
            ])
        )
        self._validate_resolutions(data_resolutions)

        resolved_config = config.resolve_resample(
            product,
            data_resolutions,
            self._component._get_streaming_resolutions(product),
        )
        if resolved_config.resample != config.resample:
            self._logger.warning(
                f"{product.name} {data_resolutions=!r} is auto-resampled from {config.resample} to {resolved_config.resample}"
            )
        config = resolved_config

        # mutually bind data_resampler and data_resamplee
        resample = cast("dict[Resolution, Resolution]", config.resample)
        for resamplee_resolution, resampler_resolution in resample.items():
            # NOTE: the resolution types are validated in DataConfig.validate_resample()
            data_resamplee = cast(
                "BarData", self._add_data(product, resamplee_resolution, config)
            )
            data_resampler = cast(
                "TickData | BarData",
                self._add_data(product, resampler_resolution, config),
            )
            # raises if the resampler can't be resampled into the resamplee
            data_resamplee.bind_resampler(data_resampler)
            self._logger.debug(
                f"{product.name} resolution={resampler_resolution} is used to resample {resamplee_resolution} data"
            )

        datas: list[MarketData] = [
            self._add_data(product, resolution, config)
            for resolution in data_resolutions
        ]
        return datas

    def create_data(
        self,
        product: BaseProduct,
        resolution: Resolution,
        config: DataConfig,
    ) -> MarketData:
        if resolution.is_quote():
            DataClass = QuoteData
        elif resolution.is_tick():
            DataClass = TickData
        elif resolution.is_bar():
            DataClass = BarData
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")
        return DataClass(product=product, resolution=resolution, config=config)

    # FIXME
    def update_quote(self, update: Quote):
        data = self.get_data(update["product"], update["resolution"])
        if data is None:
            return
        data.on_update(update["bids"], update["asks"], update["ts"], **update["extra"])
        self._databoy._deliver(data)

    # FIXME
    def update_tick(self, update: Tick):
        data = self.get_data(update["product"], update["resolution"])
        if data is None:
            return
        data.on_update(
            price=update["price"],
            volume=update["volume"],
            ts=update["ts"],
            msg_ts=update["msg_ts"],
            extra=update["extra"],
        )
        self._databoy._deliver(data)
        if data_resamplees := data.get_resamplees():
            price = update.pop("price")
            volume = update.pop("volume")
            bar_update = update.copy()
            bar_update["is_incremental"] = True
            bar_update["open"] = price
            bar_update["high"] = price
            bar_update["low"] = price
            bar_update["close"] = price
            bar_update["volume"] = volume
            for data_resamplee in data_resamplees:
                bar_update["resolution"] = data_resamplee.resolution
                self.update_bar(bar_update)

    def update_bar(self, update: Bar):
        """update bar data from streaming message
        if ready, deliver the bar data to databoy
        """

        def _update_data(data: MarketData):
            data.on_update(
                o=update["open"],
                h=update["high"],
                l=update["low"],
                c=update["close"],
                v=update["volume"],
                ts=update["ts"],
                msg_ts=update["msg_ts"],
                extra=update["extra"],
                is_incremental=update["is_incremental"],
            )

        data = self.get_data(update["product"], update["resolution"])
        # NOTE: when not using zeromq, ALL msgs are passed to all components.
        # If this component didn't subscribe to this data (via add_data()), get_data() returns None and the update is skipped.
        if data is None:
            return
        if not update["is_incremental"]:
            _update_data(data)
            self._databoy._deliver(data)
            # NOTE: in case update['ts'] < data.end_ts but the bar is already closed
            # pick the max value so that resamplees can use the correct ts to determine if they are closed
            # e.g. data is 1m bar, resamplee is 15m bar
            # without this, resamplee might not know the bar is closed when update['ts'] < data.end_ts
            update["ts"] = max(update["ts"], data.end_ts)
        else:
            # deliver the closed bar before update() clears it for the next bar.
            # NOTE: `not data.is_closed()` guard is necessary because data could be already closed after an non-incremental update
            was_open = not data.is_closed()
            has_now_reached_end = was_open and data.is_closed(
                now=update["ts"] or update["msg_ts"] or time.time()
            )
            if was_open and has_now_reached_end:
                self._databoy._deliver(data)
                _update_data(data)
            else:
                _update_data(data)
                if data.config.push_incomplete_bar:
                    self._databoy._deliver(data)

        # update resamplees
        if data_resamplees := data.get_resamplees():
            resamplee_update = update.copy()
            resamplee_update["is_incremental"] = True
            for data_resamplee in data_resamplees:
                resamplee_update["resolution"] = data_resamplee.resolution
                self.update_bar(resamplee_update)

    # TODO
    def update_df(self, data: MarketData):
        print("***update_df", data)
        print("***update_df", self._df)

    def _should_cache_resampled(self, feed: MarketFeed) -> bool:
        """Determine if the retrieved data should be cached to the CURATED layer."""
        engine_settings = self._component.context.settings
        setting = engine_settings.cache_materialized_data
        if setting is True:
            return True
        if setting is False:
            return False
        # 'auto': cache only when resampling actually occurred
        request = feed._get_current_request()
        return request.data_resolution != request.target_resolution

    def materialize(self):
        """Materializes market data by loading from storage, with optional auto-download fallback.

        For each registered data feed, first checks the cache for previously resampled data.
        If not cached, retrieves from pfeed's data lakehouse, optionally caching the result.
        If data is not found and `auto_download_data` is enabled in settings, downloads it from source.
        Missing dates in partially available data are reported as warnings but not auto-filled.

        Raises:
            DataNotFoundError: If no data is found and auto-download is disabled,
                or if the data source is paid-by-usage (auto-download of paid data is not allowed).
        """
        # A model/feature may derive its spine from same-resolution children and
        # therefore have no market data store entries of its own.
        if not self.get_datas():
            self._df = None
            return

        from pfeed.errors import DataNotFoundError

        dfs: list[nw.DataFrame[Any]] = []
        engine_context = self._component.context
        settings = engine_context.settings
        start_date, end_date = engine_context.data_start, engine_context.data_end

        component = self._component
        primary_resolution = component.resolution
        for product in component.products.values():
            # only materialize data with primary resolution
            data = self.get_data(product.name, primary_resolution)
            if data is None:
                raise Exception(
                    f"{product.name} {primary_resolution} data has not been added"
                )
            # pfund only supports bar data as the main resolution
            if not data.is_bar():
                continue
            feed = self._create_feed(data)
            self._logger.debug(
                f"Materializing market data {data.product.name} {data.resolution}..."
            )
            data_config = data.config
            storage_config = data_config.storage_config
            io_config = data_config.io_config
            product, symbol = str(data.product.basis), data.product.symbol
            product_specs = data.product.specs
            cache_storage_config = self._create_cache_storage_config(storage_config)
            requires_resampling = data.resolution in data_config.resample
            retrieve = partial(
                feed.retrieve,
                env=Environment.BACKTEST,
                product=product,
                resolution=data.resolution,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_origin=data.origin,
                dataflow_per_date=True
                if requires_resampling and data_config.num_batch_workers
                else False,
                **product_specs,
            )

            # check cache first for previously resampled data
            if settings.cache_materialized_data is not False:
                result: RunResult = retrieve(
                    storage_config=cache_storage_config,
                    io_config=io_config,
                ).run()
                # cache is best-effort: a miss/failure just falls through to original storage
                if result.failed:
                    self._logger.debug(
                        f"failed to load cached data for {data.product.name} {data.resolution}: {result.errors}"
                    )
                _df: IntoDataFrame | None = result.data
                if _df is not None:
                    self._logger.debug(
                        f"loaded data from {storage_config.data_path} for {data.product.name} {data.resolution}"
                    )
                    dfs.append(self._standardize_df(_df))
                    continue

            # cache miss or caching disabled, retrieve from original storage
            result: RunResult = (
                retrieve(storage_config=storage_config)
                .load(
                    storage_config=(
                        cache_storage_config
                        if self._should_cache_resampled(feed)
                        else None
                    ),
                    io_config=io_config,
                )
                .run()
            )
            # not critical here: a None result falls through to auto-download below
            if result.failed:
                self._logger.warning(
                    f"failed to retrieve data for {data.product.name} {data.resolution} from {storage_config.data_path}: {result.errors}"
                )
            _df: IntoDataFrame | None = result.data
            if _df is not None:
                dfs.append(self._standardize_df(_df))
                self._logger.debug(
                    f"loaded data from {storage_config.data_path} for {data.product.name} {data.resolution}"
                )
            else:
                if settings.auto_download_data:
                    # PAID data cannot be downloaded automatically, user must download it manually
                    if (
                        feed.data_source.METADATA.access_type
                        == DataAccessType.PAID_BY_USAGE
                    ):
                        raise DataNotFoundError(
                            f"No data found for {data.product.name} {data.resolution}, and auto-downloading PAID data from {feed.data_source.name} is NOT allowed"
                        )

                    self._logger.warning(
                        f"No data found for {data.product.name} {data.resolution}, auto-downloading data..."
                    )
                    result: RunResult = feed.download(
                        product=product,
                        resolution=data.resolution,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        data_origin=data.origin,
                        storage_config=storage_config,
                        io_config=io_config,
                        **product_specs,
                    ).run()
                    # critical: data is required here, so a failure must raise
                    _df: IntoDataFrame | None = result.data
                    if _df is None:
                        raise DataNotFoundError(
                            f"Failed to download data for {data.product.name} {data.resolution}"
                            + (f": {result.errors}" if result.errors else "")
                        )
                    else:
                        dfs.append(self._standardize_df(_df))
                else:
                    raise DataNotFoundError(
                        f"No data found for {data.product.name} {data.resolution}.\n"
                        + "and 'auto_download_data' is disabled in settings, please enable it in engine settings or use 'pfeed' to download the data manually."
                    )

        # concat just stacks per-(product, resolution) frames; sort by KEY_COLS
        # (date-leading) so _df holds a deterministic, date-ascending invariant
        # for every downstream consumer (get_df, merge_data_dfs, features_df)
        df = nw.concat(dfs).sort(KEY_COLS)
        if isinstance(df, nw.LazyFrame):
            df = df.collect()

        validate_spine_df(df)
        self._df = reorder_key_cols(df)

    # TODO:
    def flush_stale_bars(self):
        """Flush stale bars from the in-memory buffer."""
        pass

    # TODO:
    def rehydrate_from_lakehouse(self):
        """Load data from pfeed's data lakehouse when:
        - theres missing data, load streaming data from pfeed (data has been recorded in pfeed)
        - EOD data is available, replace SourceType.STREAM (live data) with SourceType.BATCH (official EOD data)
        """
        pass
