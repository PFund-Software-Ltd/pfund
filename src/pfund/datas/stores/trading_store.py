# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false, reportConstantRedefinition=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from pfeed.sources.pfund.component_feed import PFundComponentFeed
    from pfeed.dataflow.result import RunResult

    from pfund.datas import BarData
    from pfund.typing import Component, Signals, ColumnName

import narwhals as nw
from pfeed.storages.storage_config import StorageConfig

from pfund.enums import ArtifactType


class TradingStore:
    INDEX_COL: ClassVar[str] = "date"
    # NOTE: str(resolution) is used as values, not repr(resolution)
    PIVOT_COLS: ClassVar[list[str]] = ["product", "resolution"]

    def __init__(self, component: Component):
        import pfeed as pe

        self._component: Component = component
        # trading_df = [features_df] + [signals_df], in the component's df_form:
        # the features part is seeded by materialize() (child component signals
        # and/or the component's own data); the signals part (usually one column)
        # starts empty and is filled by update_df()
        self._df: nw.DataFrame[Any] | None = None
        self._feed: PFundComponentFeed = pe.PFund(
            pipeline_mode=True
        ).component_feed.with_component(self._component)
        self._features: dict[
            ColumnName, Any
        ] = {}  # child components signals = component's feature columns
        self._storage_config: StorageConfig | None = None

    @property
    def KEY_COLS(self) -> list[str]:
        return [self.INDEX_COL] + self.PIVOT_COLS

    @property
    def logger(self):
        return self._component.logger

    @property
    def storage_config(self) -> StorageConfig:
        assert self._storage_config is not None
        return self._storage_config

    def set_pivot_cols(self, pivot_cols: list[str]):
        self.PIVOT_COLS = pivot_cols

    def set_storage_config(self, storage_config: StorageConfig):
        self._storage_config = storage_config

    def pivot_df(self, df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        """Pivots signals dataframe from long form to wide form.

        Args:
            df: signals_df in long form
        """
        from pfund.utils.dataframe import pivot_long_to_wide

        return pivot_long_to_wide(
            df,
            index_col=self.INDEX_COL,
            pivot_cols=self.PIVOT_COLS,
        )

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
                storage convention (see MarketDataStore.adjust_date_to_bar_close_time).
        """
        import numpy as np

        component = self._component
        if component.df_form != "wide":
            raise NotImplementedError(
                f"update_df() does not support df_form='{component.df_form}' yet"
            )
        date = data.start_dt

        # live trading contract: _forward() outputs the LATEST signals — one value
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

    def materialize(self):
        """Loads child components signals stored in pfeed's data lakehouse as this component's features/factors"""
        component = self._component
        data_dfs = {
            category: data_store.get_df(to_native=True)
            for category, data_store in component.data_stores.items()
            if data_store.data_as_features and data_store.get_datas()
        }
        if data_dfs:
            self._df = nw.from_native(component.merge_data_dfs(data_dfs))
        # TODO: load features_df

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
