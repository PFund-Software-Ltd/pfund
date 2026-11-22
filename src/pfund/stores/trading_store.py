from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from pfeed.io.blob_io import BlobIO
    from pfeed.io.table_io import TableIO, VacuumResult
    from pfund_blocks.datas import BarData

    from pfund._pfeed_source.component.feed import PFundComponentFeed
    from pfund.typing import ColumnName, Component, Signals

import datetime

import polars as pl

from pfund.components.bar_dataframe import KEY_COLS
from pfund.enums import ArtifactType


class TradingStore:
    # the dir of the component's datalake in its component dir, named after what it holds, not the IO type
    TRADING_DF_DIR_NAME: ClassVar[str] = "trading_df"

    def __init__(self, component: Component):
        import pfeed as pe

        self._component: Component = component
        # trading_df = [features_df] + [signals_df], in the component's df_form:
        # vectorized backtesting constructs it from the component's data and
        # child signals; event-driven/live execution fills it incrementally;
        # materialize() rehydrates the component's own persisted trading_df.
        self._df: pl.DataFrame | None = None
        self._feed: PFundComponentFeed = pe.PFund().component_feed
        # child components signals = component's feature columns
        self._features: dict[ColumnName, Any] = {}
        self._datalake: TableIO | None = None
        self._blob_io: BlobIO | None = None

    @property
    def logger(self):
        return self._component.logger

    @property
    def datalake(self) -> TableIO:
        assert self._datalake is not None, "datalake has not been initialized"
        return self._datalake

    @property
    def blob_io(self) -> BlobIO:
        assert self._blob_io is not None, "blob_io has not been initialized"
        return self._blob_io

    def setup_lakehouse(self) -> None:
        """Sets up the lakehouse's IOs: the component's datalake (table IO) and the blob IO.

        Called after component_id is final, it is in the datalake's path:
        - <lakehouse_path>/<PFundBaseDataModel.RUNS_DIR_NAME>: the blob IO, with plain dataset dirs, i.e. runs/BACKTEST/...
        - <lakehouse_path>/<component_path>/<TRADING_DF_DIR_NAME>: the component's datalake, inside its component dir
        lakehouse_path is pfund's engine-scoped data_path by default,
        or <datalake template's base_path>/<engine name> when the user passes a datalake.
        """
        import posixpath

        from pfeed.io.blob_io import BlobIO

        from pfund._pfeed_source.base.data_model import PFundBaseDataModel

        context = self._component.context
        datalake = self._component.settings.datalake
        if datalake is None:
            import pfeed as pe

            DataLakeIO = pe.DuckLakeIO
            options = {}
            # pfund's data_path is already engine-scoped
            lakehouse_path = str(context.config.data_path)
        else:
            import inspect

            # the user's IO is only a template: it's built before any engine exists, so its
            # base_path is the lakehouse root, nest it under the engine name (like config.data_path)
            DataLakeIO = type(datalake)
            options = {
                name: getattr(datalake, f"_{name}")
                for name in inspect.signature(DataLakeIO.__init__).parameters
                if name not in ("self", "base_path")
            }
            # NOTE: posixpath, not Path, which collapses the "//" of a URI (e.g. s3://)
            lakehouse_path = posixpath.join(datalake._base_path, context.name)
        component_path = self._feed.create_data_model(
            self._component, ArtifactType.data
        ).component_path
        self._datalake = DataLakeIO(
            base_path=posixpath.join(
                lakehouse_path, component_path.as_posix(), self.TRADING_DF_DIR_NAME
            ),
            **options,
        )
        self._blob_io = BlobIO(
            base_path=posixpath.join(lakehouse_path, PFundBaseDataModel.RUNS_DIR_NAME),
            hive_dataset_dirs=False,
        )

    def has_updated(self, data: BarData) -> bool:
        """Return whether the trading dataframe already contains ``data``'s bar.

        Long-form dataframes preserve the complete bar key
        ``(date, product, resolution)``. Wide-form dataframes fold ``product``
        and ``resolution`` into value-column names, so their row key is only
        ``date``.
        """
        df = self._df
        if df is None or df.is_empty():
            return False

        component = self._component
        if component.df_form == "long":
            predicate = (
                (pl.col(self.INDEX_COL) == data.start_dt)
                & (pl.col("product") == data.product.name)
                & (pl.col("resolution") == str(data.resolution))
            )
        elif component.df_form == "wide":
            predicate = pl.col(self.INDEX_COL) == data.start_dt
        else:
            raise ValueError(f"unsupported dataframe form {component.df_form!r}")

        return not df.filter(predicate).is_empty()

    def get_df(
        self,
        kind: Literal["features", "signals", "trading"] = "trading",
        window_size: int | None = None,
    ) -> pl.DataFrame:
        """
        Args:
            kind: Which part of the trading_df to return.
                - "features": the features used to compute signals
                - "signals": the computed signals
                - "trading": the full trading DataFrame
            window_size: Number of most recent rows to return.
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
                        f"{component.name} has no signals or signal_cols is empty ({signal_cols=}) in its trading df yet"
                    )
                # only KEY_COLS actually present are real keys (wide form folds pivot cols into value-column names)
                key_cols = [col for col in KEY_COLS if col in columns]
                df = df.select(key_cols + signal_cols)
            else:  # features
                df = df.select([col for col in columns if col not in signal_cols])
        if window_size is not None:
            df = df.tail(window_size)
        return df

    def get_signals(self, data: BarData) -> Signals | None:
        df = self._df
        if df is None:
            return None

        if self._component.df_form == "long":
            predicate = (
                (pl.col("date") == data.start_dt)
                & (pl.col("product") == data.product.name)
                & (pl.col("resolution") == str(data.resolution))
            )
        else:
            predicate = pl.col("date") == data.start_dt

        row = df.filter(predicate).head(1)
        if row.is_empty():
            return None

        signal_cols = [
            col for col in self._component._signal_cols if col in row.columns
        ]
        if not signal_cols:
            return None

        return {col: row.item(0, col) for col in signal_cols}

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
        # A trading dataframe may only grow on its owning component's bar
        # cadence. For example, a 1d child signal cannot create a row in a 1m
        # model's dataframe; it must be as-of aligned when a legitimate 1m bar
        # drives the model update.
        if not data.resolution.is_strict_equal(component.resolution):
            raise ValueError(
                f"{component.name} cannot update its {component.resolution!r} "
                + f"trading dataframe from a {data.resolution!r} bar"
            )
        if component.df_form != "wide":
            raise NotImplementedError(
                f"update_df() does not support df_form='{component.df_form}' yet"
            )
        date = data.start_dt

        # live trading contract: step() outputs the LATEST signals — one value
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
            df = pl.DataFrame(
                {self.INDEX_COL: [date], **{col: [v] for col, v in row_values.items()}}
            )
        else:
            index_col = self.INDEX_COL
            row_exists = not df.filter(pl.col(index_col) == date).is_empty()
            if row_exists:
                df = df.with_columns(
                    pl.when(pl.col(index_col) == date)
                    .then(pl.lit(v))
                    .otherwise(pl.col(col) if col in df.columns else pl.lit(None))
                    .alias(col)
                    for col, v in row_values.items()
                )
            else:
                new_row = pl.DataFrame(
                    {index_col: [date], **{col: [v] for col, v in row_values.items()}}
                )
                # diagonal: the new row only has [date] + this contributor's columns;
                # every other column is filled with null
                df = pl.concat([df, new_row], how="diagonal")

        max_rows = component.config["max_rows"]
        if max_rows is not None and len(df) > max_rows:
            df = df.tail(max_rows)
        self._df = df

    def _save_source_artifact(self) -> None:
        self._feed.save_artifact(self._component, ArtifactType.source)

    # TODO: remove strategy columns, e.g. "signal", "order_xxx" and "trade_xxx"
    # they are unnecessary and could be misleading (if created by vectorized backtesting, results are inaccurate)
    def materialize(self):
        """Load this component's own trading_df.delta (data artifact)."""
        data = self._feed.load_artifact(self._component, ArtifactType.data)
        if data is None:
            self._df = None
            self.logger.debug(
                f"No persisted trading dataframe found for '{self._component.name}'"
            )
            return

        assert isinstance(data, pl.LazyFrame), (
            f"data artifact must be a polars LazyFrame, got {type(data)}"
        )
        self._df = data.collect()

        metadata = self._feed.load_metadata(self._component)
        if metadata and metadata["signal_cols"]:
            self._component.set_signal_cols(metadata["signal_cols"])

    def persist_to_datalake(self) -> None:
        """Persist the component's current data artifact through pfeed.

        The dataframe is not passed directly to pfeed. The component feed treats
        the bound component as a data source and extracts ``component._data_artifact``
        when it saves.

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
            return
        self._feed.save_artifact(component, ArtifactType.data)

    def optimize_datalake(self) -> None:
        """Compacts the datalake's tables, see ``TableIO.optimize()``."""
        self.datalake.optimize()

    def vacuum_datalake(
        self, retention_hours: int | None = None, dry_run: bool = True
    ) -> VacuumResult:
        """Deletes the datalake's history older than ``retention_hours``, see ``TableIO.vacuum()``.

        Pfeed's safe defaults apply: ``dry_run=True`` and ``TableIO.DEFAULT_VACUUM_RETENTION``.
        """
        if retention_hours is None:
            return self.datalake.vacuum(dry_run=dry_run)
        return self.datalake.vacuum(
            retention=datetime.timedelta(hours=retention_hours), dry_run=dry_run
        )
