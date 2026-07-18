# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false, reportConstantRedefinition=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from datetime import timedelta

    from deltalake.table import FilterConjunctionType
    from deltalake.transaction import CommitProperties, PostCommitHookProperties
    from deltalake.writer.properties import WriterProperties
    from narwhals.typing import IntoDataFrame
    from pfeed.sources.pfund.component_feed import PFundComponentFeed
    from pfeed.dataflow.result import RunResult
    from pfeed.storages.base_storage import BaseStorage

    from pfund.datas import BarData
    from pfund.typing import Component, Signals, ColumnName

import narwhals as nw
from pfeed.storages.storage_config import StorageConfig

from pfund.enums import ArtifactType


class TradingStore:
    def __init__(self, component: Component):
        import pfeed as pe

        self._component: Component = component
        # trading_df = [features_df] + [signals_df], in the component's df_form:
        # vectorized backtesting constructs it from the component's data and
        # child signals; event-driven/live execution fills it incrementally;
        # materialize() rehydrates the component's own persisted trading_df.
        self._df: nw.DataFrame[Any] | None = None
        self._feed: PFundComponentFeed = pe.PFund(
            pipeline_mode=True
        ).component_feed.with_component(self._component)
        self._features: dict[
            ColumnName, Any
        ] = {}  # child components signals = component's feature columns
        self._storage_config: StorageConfig | None = None
        self._lakehouse_storage: BaseStorage | None = None

    @property
    def logger(self):
        return self._component.logger

    @property
    def storage_config(self) -> StorageConfig:
        assert self._storage_config is not None
        return self._storage_config

    def set_lakehouse_storage(self, storage_config: StorageConfig):
        self._storage_config = storage_config
        Storage = storage_config.storage.storage_class
        io_config = self._feed._create_artifact_io_config(ArtifactType.data)
        self._lakehouse_storage = (
            Storage.from_storage_config(storage_config)
            .with_io(io_config)
            .with_data_model(
                self._feed.create_data_model(artifact_type=ArtifactType.data)
            )
        )

    def has_updated(self, data: BarData) -> bool:
        """Return whether the trading dataframe already contains ``data``'s bar.

        Long-form dataframes preserve the complete bar key
        ``(date, product, resolution)``. Wide-form dataframes fold ``product``
        and ``resolution`` into value-column names, so their row key is only
        ``date``.
        """
        df = self._df
        if df is None or len(df) == 0:
            return False

        component = self._component
        if component.df_form == "long":
            predicate = (
                (nw.col(self.INDEX_COL) == data.start_dt)
                & (nw.col("product") == data.product.name)
                & (nw.col("resolution") == str(data.resolution))
            )
        elif component.df_form == "wide":
            predicate = nw.col(self.INDEX_COL) == data.start_dt
        else:
            raise ValueError(f"unsupported dataframe form {component.df_form!r}")

        return len(df.filter(predicate).head(1)) > 0

    def get_df(
        self,
        kind: Literal["features", "signals", "trading"] = "trading",
        window_size: int | None = None,
        to_native: bool = False,
    ) -> nw.DataFrame[Any] | IntoDataFrame:
        """
        Args:
            kind: Which part of the trading_df to return.
                - "features": the features used to compute signals
                - "signals": the computed signals
                - "trading": the full trading DataFrame
            window_size: Number of most recent rows to return.
            to_native: If True, return the underlying backend frame (polars/pandas) instead
                of a Narwhals DataFrame. Defaults to True.
        """
        df = self._df
        component = self._component
        if df is None:
            raise RuntimeError(f"{component.name} trading df is not ready")
        if kind != "trading":
            columns = df.columns
            # signal cols are locked on first signalize(); intersect with columns because they
            # may be locked but not yet written (e.g. features materialized, _forward not run yet)
            signal_cols = [col for col in component._signal_cols if col in columns]
            if kind == "signals":
                if not signal_cols:
                    raise RuntimeError(
                        f"{component.name} has no signals in its trading df yet"
                    )
                # only KEY_COLS actually present are real keys (wide form folds pivot cols into value-column names)
                key_cols = [col for col in self.KEY_COLS if col in columns]
                df = df.select(key_cols + signal_cols)
            else:  # features
                df = df.select([col for col in columns if col not in signal_cols])
        if window_size is not None:
            df = df.tail(window_size)
        return df.to_native() if to_native else df

    # TODO: need to handle if data_as_features is True, i.e. if data is also a feature
    # TODO: check if the data.is_closed() is False (incomplete bar)
    #   if thats true, it means user manually calls forward(), need to mark it in the df
    def update_df(self, signals: Signals, data: BarData):
        """Upserts signals into the current bar's row of the trading df.

        Called once per contributor per closed bar — each child component's signals
        (this component's features), then this component's own signals. Whichever
        contributor arrives first creates the bar's row; later contributors fill
        their named columns on that same row. Keyed by column name, hence
        *update*, not append.

        Args:
            signals: a child component's signals or this component's signals
            data: the bar the signals were computed for. The store derives the
                row key from it: the bar's open time, matching the market data
                storage convention. Close-time availability is used only as a
                temporary alignment boundary in the bar-dataframe utilities.
        """
        import numpy as np

        component = self._component
        if component.df_form != "wide":
            raise NotImplementedError(
                f"update_df() does not support df_form='{component.df_form}' yet"
            )
        date = data.start_dt

        # live trading contract: forward() outputs the LATEST signals — one value
        # per signal column for the current bar (a lookback window is input context,
        # not output grain)
        row_values: dict[ColumnName, Any] = {}
        for col, value in signals.items():
            value = np.asarray(value)
            if value.size != 1:
                raise ValueError(
                    f"{component.name} got {value.size} values for signal column '{col}' "
                    + "on a single bar; in live trading, transform()/predict()/decide() must "
                    + "output only the latest signals (one value per signal column)"
                )
            row_values[col] = value.item()

        df = self._df
        if df is None:
            # no seeded features (component without data-as-features): the first
            # contributor's signals create the trading df
            from pfeed.config import get_config

            df = nw.DataFrame.from_dict(
                {self.INDEX_COL: [date], **{col: [v] for col, v in row_values.items()}},
                backend=str(get_config().data_tool),
            )
        else:
            index_col = self.INDEX_COL
            row_exists = len(df.filter(nw.col(index_col) == date)) > 0
            if row_exists:
                df = df.with_columns(
                    nw.when(nw.col(index_col) == date)
                    .then(nw.lit(v))
                    .otherwise(nw.col(col) if col in df.columns else nw.lit(None))
                    .alias(col)
                    for col, v in row_values.items()
                )
            else:
                new_row = nw.DataFrame.from_dict(
                    {index_col: [date], **{col: [v] for col, v in row_values.items()}},
                    backend=nw.get_native_namespace(df),
                )
                # diagonal: the new row only has [date] + this contributor's columns;
                # every other column is filled with null
                df = nw.concat([df, new_row], how="diagonal")

        max_rows = component.config["max_rows"]
        if max_rows is not None and len(df) > max_rows:
            df = df.tail(max_rows)
        self._df = df

    def _save_source_artifact(self) -> RunResult:
        return self._feed.download(
            artifact_type=ArtifactType.source,
            storage_config=self.storage_config,
        ).run()

    # TODO: remove all the "order_xxx" and "trade_xxx" columns
    def materialize(self) -> bool:
        """Load this component's own persisted trading dataframe.

        Returns:
            Whether a persisted trading dataframe was found.
        """
        result = self._feed.retrieve(
            artifact_type=ArtifactType.data,
            storage_config=self.storage_config,
        ).run()
        if result.data is None:
            self._df = None
            self.logger.debug(
                f"No persisted trading dataframe found for '{self._component.name}'"
            )
            return False

        df = nw.from_native(cast("IntoDataFrame", result.data))
        self._df = df.collect() if isinstance(df, nw.LazyFrame) else df
        return True

    def persist_to_lakehouse(self) -> RunResult | None:
        """Persist the component's current data artifact through pfeed.

        The dataframe is not passed directly to pfeed. The component feed treats
        the bound component as a data source and extracts ``component._data_artifact``
        when the dataflow runs.

        Which rows are exposed by ``data_artifact`` depends on the execution mode:
        vectorized backtesting exposes the complete trading dataframe, while
        event-driven and live execution will expose the rows selected by
        TradingStore for the current persistence interval.
        """
        # nothing to persist yet (e.g. component still warming up)
        component = self._component
        has_signals = self._df is not None and any(
            col in self._df.columns for col in component._signal_cols
        )
        if not has_signals:
            self.logger.debug(f"{component.name} has no signals to persist yet")
            return None
        return self._feed.download(
            artifact_type=ArtifactType.data,
            storage_config=self.storage_config,
        ).run()

    def optimize_lakehouse(
        self,
        partition_filters: FilterConjunctionType | None = None,
        target_size: int | None = None,
        max_concurrent_tasks: int | None = None,
        max_spill_size: int | None = None,
        max_temp_directory_size: int | None = None,
        min_commit_interval: int | timedelta | None = None,
        writer_properties: WriterProperties | None = None,
        post_commithook_properties: PostCommitHookProperties | None = None,
        commit_properties: CommitProperties | None = None,
    ) -> dict[str, Any] | None:
        """Compact this component's Delta table using pfeed's implementation.

        Returns ``None`` until the component has successfully persisted its first
        trading dataframe.
        """
        storage = self._lakehouse_storage
        if storage is None:
            self.logger.debug(
                f"{self._component.name} has no Delta table to optimize yet"
            )
            return None
        from pfeed.storages.deltalake_storage_mixin import DeltaLakeStorageMixin

        if not isinstance(storage, DeltaLakeStorageMixin):
            raise TypeError(f"{storage.name} does not support optimize_lakehouse")
        from deltalake.exceptions import TableNotFoundError

        try:
            delta_table = storage.get_delta_table()
        except TableNotFoundError:
            self.logger.debug(
                f"{self._component.name} has no Delta table to optimize yet"
            )
            return None
        return storage.optimize_delta_table(
            delta_table,
            partition_filters=partition_filters,
            target_size=target_size,
            max_concurrent_tasks=max_concurrent_tasks,
            max_spill_size=max_spill_size,
            max_temp_directory_size=max_temp_directory_size,
            min_commit_interval=min_commit_interval,
            writer_properties=writer_properties,
            post_commithook_properties=post_commithook_properties,
            commit_properties=commit_properties,
        )

    def vacuum_lakehouse(
        self,
        retention_hours: int | None = None,
        dry_run: bool = True,
        enforce_retention_duration: bool = True,
        post_commithook_properties: PostCommitHookProperties | None = None,
        commit_properties: CommitProperties | None = None,
        full: bool = False,
        keep_versions: list[int] | None = None,
    ) -> list[str] | None:
        """Vacuum this component's Delta table using pfeed's implementation.

        Pfeed's safe defaults apply, including ``dry_run=True``. Returns ``None``
        until the component has successfully persisted its first trading
        dataframe.
        """
        storage = self._lakehouse_storage
        if storage is None:
            self.logger.debug(
                f"{self._component.name} has no Delta table to vacuum yet"
            )
            return None
        from pfeed.storages.deltalake_storage_mixin import DeltaLakeStorageMixin

        if not isinstance(storage, DeltaLakeStorageMixin):
            raise TypeError(f"{storage.name} does not support vacuum_lakehouse")
        from deltalake.exceptions import TableNotFoundError

        try:
            delta_table = storage.get_delta_table()
        except TableNotFoundError:
            self.logger.debug(
                f"{self._component.name} has no Delta table to vacuum yet"
            )
            return None
        return storage.vacuum_delta_table(
            delta_table,
            retention_hours=retention_hours,
            dry_run=dry_run,
            enforce_retention_duration=enforce_retention_duration,
            post_commithook_properties=post_commithook_properties,
            commit_properties=commit_properties,
            full=full,
            keep_versions=keep_versions,
        )
