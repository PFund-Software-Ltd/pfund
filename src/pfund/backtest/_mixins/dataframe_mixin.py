from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from pfund.backtest.cv.cv_fold import CVFold
    from pfund.backtest.cv.holdout_split import HoldoutSplit
    from pfund.typing import Component

import narwhals as nw
from pfeed.enums import DataCategory

from pfund.backtest.cv.indexing import SplittableData
from pfund.components._mixins.dataframe_mixin import DataFrameMixin
from pfund.components.bar_dataframe import (
    INDEX_COL as BAR_INDEX_COL,
    KEY_COLS as BAR_KEY_COLS,
    PIVOT_COLS as BAR_PIVOT_COLS,
    align_df_to_spine,
    pivot_long_to_wide,
    reorder_key_cols,
    validate_spine_df,
)


class BacktestDataFrameMixin(DataFrameMixin):
    @property
    def folds(self) -> tuple[CVFold, ...]:
        return self._folds

    def _resolve_dataset_splits(self) -> None:
        from pfund.backtest.cv.cross_validation import resolve_folds
        from pfund.backtest.cv.holdout import Holdout, resolve_holdout
        from pfund.backtest.cv.holdout_split import HoldoutSplit

        splits = self.context.dataset_splits
        if splits is None:
            self._holdout_split, self._folds = HoldoutSplit(self.X), ()
        elif isinstance(splits, Holdout):
            self._holdout_split, self._folds = resolve_holdout(self.X, splits), ()
        else:
            self._holdout_split, self._folds = None, resolve_folds(self.X, splits)

    def _get_holdout_split(self) -> HoldoutSplit | None:
        if self._holdout_split is None and not self._folds:
            self._resolve_dataset_splits()
        return self._holdout_split

    def split[DataT: SplittableData](
        self,
        data: DataT,
    ) -> tuple[DataT | None, DataT | None, DataT | None]:
        """Split aligned data into the component's train/validation/test sets.

        `data` is sliced positionally by selectors built from X, so it needs
        no `date` column of its own — it only has to be row-aligned 1:1 with
        X (same length and order). A target y, class labels, sample
        weights, or any per-row companion of the features therefore uses the
        same boundaries as X.

            close = next(c for c in model.df.columns if c.endswith(":close"))
            y = model.df.get_column(close).shift(-1) / model.df.get_column(close) - 1.0
            X_train, X_val, _ = model.split(model.X)
            y_train, y_val, _ = model.split(y)
            model.fit(X_train, y_train)

        Args:
            data: a DataFrame, Series, or ndarray row-aligned 1:1 with X.

        Returns a (train, val, test) tuple; any segment that collapsed to 0
        timestamps or is undefined for the split is None.

        Cross-validation has multiple splits, so call ``fold.split(data)`` on
        each item in ``component.folds`` instead.
        """
        holdout_split = self._get_holdout_split()
        if holdout_split is None:
            raise ValueError(
                "component.split() is undefined under cross-validation; "
                + "iterate over component.folds and call fold.split(data)"
            )
        return holdout_split.split(data)

    @property
    def X_train(self) -> IntoDataFrame | None:
        """Feature rows in the single training split.

        Cross-validation has multiple training splits, so access them through
        ``fold.X_train`` instead.
        """
        holdout_split = self._get_holdout_split()
        return None if holdout_split is None else holdout_split.X_train

    @property
    def X_val(self) -> IntoDataFrame | None:
        """Feature rows in the single validation split.

        Cross-validation has multiple validation splits, so access them through
        ``fold.X_val`` instead.
        """
        holdout_split = self._get_holdout_split()
        return None if holdout_split is None else holdout_split.X_val

    @property
    def X_test(self) -> IntoDataFrame | None:
        """Feature rows in the single test split.

        Cross-validation does not define a test split.
        """
        holdout_split = self._get_holdout_split()
        return None if holdout_split is None else holdout_split.X_test

    def _merge_signals_dfs(self) -> nw.DataFrame[Any]:
        """Build this component's historical feature frame.

        The component's own data establishes its receiving bar spine when
        present. Otherwise, same-resolution child signals establish the spine.
        Each child component's signal dataframe is then aligned to that spine
        and merged as feature columns. Once established, child signals can add
        values but cannot add rows to the receiving universe.

        Strictly equal resolutions use exact alignment; other resolutions use
        close-time as-of alignment. Long-form child signals are aligned per
        product. Wide-form child signals are aligned as one cross-sectional
        date stream and are broadcast by date when the receiver is long-form.
        """

        def _align_wide_child_signals(
            base_df: nw.DataFrame[Any],
            child: Component,
            child_signals: nw.DataFrame[Any],
            *,
            alignment_mode: Literal["exact", "asof"],
        ) -> nw.DataFrame[Any]:
            # A wide dataframe is one cross-sectional stream keyed by date.
            # Add a synthetic product and the known component resolutions so
            # the canonical bar aligner can apply the same close-time rules.
            wide_stream = "__wide_stream__"
            alignment_spine = (
                base_df.select(BAR_INDEX_COL)
                .unique()
                .with_columns(
                    nw.lit(wide_stream).alias("product"),
                    nw.lit(str(self.resolution)).alias("resolution"),
                )
            )
            alignment_contributor = child_signals.with_columns(
                nw.lit(wide_stream).alias("product"),
                nw.lit(str(child.resolution)).alias("resolution"),
            )
            return align_df_to_spine(
                alignment_spine,
                alignment_contributor,
                mode=alignment_mode,
            ).drop(BAR_PIVOT_COLS)

        def _align_long_child_signals_to_wide_parent(
            base_df: nw.DataFrame[Any],
            child_signals: nw.DataFrame[Any],
            *,
            alignment_mode: Literal["exact", "asof"],
        ) -> nw.DataFrame[Any]:
            # A wide parent has no product rows on which to perform an as-of
            # join. Temporarily expand its receiving dates across the child's
            # products, align each product stream independently, then pivot the
            # aligned result back to the parent's wide form.
            child_products = child_signals.select("product").unique()
            alignment_spine = (
                base_df.select(BAR_INDEX_COL)
                .unique()
                .join(child_products, how="cross")
                .with_columns(nw.lit(str(self.resolution)).alias("resolution"))
                .select(BAR_KEY_COLS)
            )
            aligned_signals = align_df_to_spine(
                alignment_spine,
                child_signals,
                mode=alignment_mode,
            )
            return pivot_long_to_wide(
                aligned_signals,
                index_col=BAR_INDEX_COL,
                pivot_cols=BAR_PIVOT_COLS,
            )

        def _join_child_signals(
            base_df: nw.DataFrame[Any],
            child: Component,
            child_signals: nw.DataFrame[Any],
        ) -> nw.DataFrame[Any]:
            alignment_mode = (
                "exact" if child.resolution.is_strict_equal(self.resolution) else "asof"
            )
            if self.df_form == "long" and child.df_form == "long":
                join_cols = BAR_KEY_COLS
                child_signals = align_df_to_spine(
                    base_df,
                    child_signals,
                    mode=alignment_mode,
                )
            elif self.df_form == "wide" and child.df_form == "long":
                join_cols = [BAR_INDEX_COL]
                child_signals = _align_long_child_signals_to_wide_parent(
                    base_df,
                    child_signals,
                    alignment_mode=alignment_mode,
                )
            else:
                join_cols = [BAR_INDEX_COL]
                child_signals = _align_wide_child_signals(
                    base_df,
                    child,
                    child_signals,
                    alignment_mode=alignment_mode,
                )

            child_value_cols = [
                col for col in child_signals.columns if col not in join_cols
            ]
            conflicting_cols = [
                col for col in child_value_cols if col in base_df.columns
            ]
            if conflicting_cols:
                raise ValueError(
                    f"child signal columns already exist in {self.name}'s features: "
                    + f"{conflicting_cols}"
                )
            return base_df.join(
                child_signals,
                on=join_cols,
                how="left",
            ).sort(join_cols)

        children_signals = [
            (
                child,
                child.get_df(kind="signals", window_size=None, to_native=False),
            )
            for child in self.components
        ]
        data_dfs = {
            category: data_store.get_df(to_native=False)
            for category, data_store in self.data_stores.items()
            # NOTE: market data df is ALWAYS included to create the "spine" of the df
            if (category == DataCategory.MARKET_DATA or data_store.data_as_features)
            and data_store.get_datas()
        }
        if data_dfs:
            df = self._merge_data_dfs(data_dfs)
            market_data_store = self.market_data_store
            if not market_data_store.data_as_features:
                market_data_df = market_data_store.get_df(to_native=False)
                market_data_cols = [
                    col
                    for col in market_data_df.columns
                    if col not in BAR_KEY_COLS + market_data_store.METADATA_COLS
                ]
                df = df.drop(market_data_cols, strict=False)

            validate_spine_df(df)
            if self.df_form == "wide":
                value_cols = [col for col in df.columns if col not in BAR_KEY_COLS]
                if value_cols:
                    df = pivot_long_to_wide(
                        df,
                        index_col=BAR_INDEX_COL,
                        pivot_cols=BAR_PIVOT_COLS,
                    )
                else:
                    df = df.select(BAR_INDEX_COL).unique().sort(BAR_INDEX_COL)
        else:
            spine_cols = BAR_KEY_COLS if self.df_form == "long" else [BAR_INDEX_COL]
            spine_dfs = [
                child_signals.select(spine_cols)
                for child, child_signals in children_signals
                if child.resolution.is_strict_equal(self.resolution)
                and (self.df_form == "wide" or child.df_form == "long")
            ]
            if not spine_dfs:
                required_child = (
                    "same-resolution child signals"
                    if self.df_form == "wide"
                    else "same-resolution long-form child signals"
                )
                raise ValueError(
                    f"{self.name} has no data or {required_child} to establish "
                    + "its bar spine"
                )
            df = nw.concat(spine_dfs).unique().sort(spine_cols)

            if self.df_form == "long":
                validate_spine_df(df)

        for child, child_signals in children_signals:
            df = _join_child_signals(df, child, child_signals)

        return reorder_key_cols(df, df_form=self.df_form)
