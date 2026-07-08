# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportUnnecessaryComparison=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict, ClassVar

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from pfeed.dataflow.result import RunResult

    from pfund.datas.databoy import DataBoy
    from pfund.datas.resolution import Resolution
    from pfund.entities.products.product_base import BaseProduct
    from pfund.typing import ProductName

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
        extra: dict[str, Any]


import time
from collections import defaultdict
from functools import partial

import narwhals as nw

from pfeed.enums import DataAccessType, DataLayer, DataStorage
from pfeed.feeds.market_feed import MarketFeed
from pfund.datas import BarData, QuoteData, TickData
from pfund.datas.data_config import DataConfig
from pfund.datas.data_market import MarketData
from pfund.datas.resolution import Resolution
from pfund.datas.stores.base_data_store import BaseDataStore
from pfund.enums import Environment


class MarketDataStore(BaseDataStore[MarketData, MarketFeed]):
    # Columns pinned to the left side of the materialized dataframe for readability
    LEFT_COLS: ClassVar[list[str]] = ["date", "resolution", "product", "source_type"]
    PIVOT_COLS: ClassVar[list[str]] = ["resolution", "product"]
    METADATA_COLS: ClassVar[list[str]] = ["source_type"]

    def __init__(self, databoy: DataBoy):
        super().__init__(databoy)
        self._datas: dict[ProductName, dict[ResolutionRepr, MarketData]] = defaultdict(
            dict
        )

    @staticmethod
    def adjust_date_to_bar_close_time(df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        """Shift 'date' from bar open time (storage convention) to bar close time.

        Bars are stored labeled by their open time, but a bar's information only
        exists once the bar closes: a 1m bar labeled 00:00:00 is formed at
        00:00:59.999, a 15m bar at 00:14:59.999. Aligning rows across
        resolutions (and avoiding lookahead) is only correct on close time.
        Each row is shifted by its own 'resolution' column value, using the
        same convention as BarData's end_ts (start + period - 1ms).
        """
        from datetime import timedelta

        expr = nw.col("date")
        for res in df.get_column("resolution").unique().to_list():
            offset = timedelta(seconds=Resolution(res).to_seconds(), milliseconds=-1)
            expr = (
                nw.when(nw.col("resolution") == res)
                .then(nw.col("date") + offset)
                .otherwise(expr)
            )
        return df.with_columns(expr.alias("date"))

    def get_data(
        self, product: ProductName, resolution: Resolution | ResolutionRepr
    ) -> MarketData | None:
        if product not in self._datas:
            return None
        if isinstance(resolution, Resolution):
            resolution = repr(resolution)
        return self._datas[product].get(resolution, None)

    def get_datas(self) -> list[MarketData]:
        return list(
            set(
                data
                for data_per_resolution in self._datas.values()
                for data in data_per_resolution.values()
            )
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

    def _resolve_data_config(
        self,
        product: BaseProduct,
        resolutions: list[Resolution | str],
        data_config: DataConfig,
    ) -> DataConfig:
        primary_resolution = self._databoy._component.resolution
        extra_resolutions = [Resolution(r) for r in resolutions]
        data_config.data_resolutions = [primary_resolution] + extra_resolutions
        resolved_data_config = (
            data_config.auto_shift()
            .auto_resample(self._databoy._component._get_supported_resolutions(product))
            .auto_skip_first_bar()
            .auto_set_stale_bar_timeout()
        )
        if resolved_data_config.resample != data_config.resample:
            self._logger.warning(
                f"{product.name} {primary_resolution=!r} {extra_resolutions=!r} "
                + f"is auto-resampled from {data_config.resample} to {resolved_data_config.resample}"
            )
        if resolved_data_config.storage_config is None:
            from pfeed.storages.storage_config import StorageConfig

            resolved_data_config.storage_config = StorageConfig()
        if resolved_data_config.io_config is None:
            from pfeed._io.io_config import IOConfig

            resolved_data_config.io_config = IOConfig()
        return resolved_data_config

    def add_data(
        self,
        product: BaseProduct,
        resolutions: list[Resolution | str] | None = None,
        config: DataConfig | None = None,
    ) -> list[MarketData]:
        config = self._resolve_data_config(
            product=product,
            resolutions=resolutions or [],
            data_config=config or DataConfig(),
        )

        # mutually bind data_resampler and data_resamplee
        for resamplee_resolution, resampler_resolution in config.resample.items():
            data_resamplee = self._add_data(product, resamplee_resolution, config)
            data_resampler = self._add_data(product, resampler_resolution, config)
            data_resamplee.bind_resampler(data_resampler)
            self._logger.debug(
                f"{product.name} resolution={resampler_resolution} is used to resample {resamplee_resolution} data"
            )

        datas: list[MarketData] = [
            self._add_data(product, resolution, config)
            for resolution in config.data_resolutions
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

    # TODO
    def update_quote(self, update: QuoteUpdate):
        data = self.get_data(update["product"], update["resolution"])
        if data is None:
            return
        data.on_update(update["bids"], update["asks"], update["ts"], **update["extra"])
        self._databoy._deliver(data)

    # TODO
    def update_tick(self, update: TickUpdate):
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

    def update_bar(self, update: BarUpdate):
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
            if not data.is_closed() and data.is_closed(
                now=update["ts"] or update["msg_ts"] or time.time()
            ):
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
        engine_settings = self._databoy._component.context.settings
        setting = engine_settings.cache_materialized_data
        if setting is True:
            return True
        if setting is False:
            return False
        # 'auto': cache only when resampling actually occurred
        request = feed._get_current_request()
        return request.data_resolution != request.target_resolution

    def materialize(self) -> nw.DataFrame[Any]:
        """Materializes market data by loading from storage, with optional auto-download fallback.

        For each registered data feed, first checks the cache for previously resampled data.
        If not cached, retrieves from pfeed's data lakehouse, optionally caching the result.
        If data is not found and `auto_download_data` is enabled in settings, downloads it from source.
        Missing dates in partially available data are reported as warnings but not auto-filled.

        Raises:
            DataNotFoundError: If no data is found and auto-download is disabled,
                or if the data source is paid-by-usage (auto-download of paid data is not allowed).
        """
        from pfeed.errors import DataNotFoundError

        dfs: list[nw.DataFrame[Any]] = []
        engine_context = self._databoy._component.context
        settings = engine_context.settings
        start_date, end_date = engine_context.data_start, engine_context.data_end

        component = self._databoy._component
        primary_resolution = component.resolution
        for product in component.products.values():
            # REVIEW: only materialize data with primary resolution?
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
                    self._logger.info(
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
                self._logger.info(
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
        df = nw.concat(dfs).sort(self.KEY_COLS)
        if isinstance(df, nw.LazyFrame):
            df = df.collect()
        self._df = df
        cols = df.columns
        assert self.INDEX_COL in cols, (
            f"Index column {self.INDEX_COL} not found in {cols}"
        )
        assert all(col in cols for col in self.PIVOT_COLS), (
            f"Pivot columns {self.PIVOT_COLS} not found in {cols}"
        )
        return df
