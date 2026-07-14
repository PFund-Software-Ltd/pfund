from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from pfeed.storages.storage_config import StorageConfig

    from pfund.datas.databoy import DataBoy

from abc import ABC, abstractmethod

import narwhals as nw
from pfeed.enums import DataLayer, DataStorage
from pfeed.feeds.base_feed import BaseFeed

from pfund.datas.data_base import BaseData


DataT = TypeVar("DataT", bound=BaseData)
FeedT = TypeVar("FeedT", bound=BaseFeed)


def pivot_long_to_wide(
    df: nw.DataFrame[Any],
    *,
    index_col: str,
    pivot_cols: list[str],
    key_cols: list[str],
) -> nw.DataFrame[Any]:
    """Pivots a long-form dataframe to wide form with flat, reference-safe names.

    Each pivoted value column is named ``{pivot_key}:{field}``, where ``pivot_key``
    joins the ``pivot_cols`` values with ":" in ``pivot_cols`` order, e.g.
    ``1_MINUTE:BYBIT_BTC_USDT_PERPETUAL:close``. This replaces polars' default
    ``close_{"1_MINUTE","BYBIT_BTC_USDT_PERPETUAL"}`` naming, which embeds
    quotes/commas that break direct column references.

    Args:
        df: dataframe in long form
        index_col: the column to keep as the wide-form index (e.g. "date")
        pivot_cols: columns folded into the wide-form column names
        key_cols: all non-value columns (index + pivot); everything else is a value
    """
    value_cols = [col for col in df.columns if col not in key_cols]
    # collapse the multi-column key into one ":"-joined key, so polars emits
    # flat "{field}{sep}{pivot_key}" names instead of its {"a","b"} tuple form
    pivot_key_col = "__pivot_key__"
    df = df.with_columns(
        nw.concat_str(
            [nw.col(col).cast(nw.String) for col in pivot_cols],
            separator=":",
        ).alias(pivot_key_col)
    )
    pivot_keys = df.get_column(pivot_key_col).unique().to_list()
    # NUL separator: never collides with field names that contain "_"
    # (e.g. n_data_points), so the rename below is unambiguous
    sep = "\x00"
    wide = df.pivot(on=pivot_key_col, index=index_col, values=value_cols, separator=sep)
    rename = {
        f"{field}{sep}{pivot_key}": f"{pivot_key}:{field}"
        for field in value_cols
        for pivot_key in pivot_keys
    }
    return wide.rename(rename).sort(index_col)


class BaseDataStore(ABC, Generic[DataT, FeedT]):
    LEFT_COLS: ClassVar[list[str]] = []
    INDEX_COL: ClassVar[str] = "date"
    PIVOT_COLS: ClassVar[list[str]] = []
    METADATA_COLS: ClassVar[list[str]] = []

    def __init__(self, databoy: DataBoy):
        self._databoy: DataBoy = databoy
        self._logger = databoy.logger
        self._df: nw.DataFrame[Any] | None = None  # in long form
        self._data_as_features: bool | None = None

    @property
    def KEY_COLS(self) -> list[str]:
        return [self.INDEX_COL] + self.PIVOT_COLS

    @abstractmethod
    def materialize(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def create_data(self, *args: Any, **kwargs: Any) -> DataT:
        pass

    @abstractmethod
    def add_data(self, *args: Any, **kwargs: Any) -> list[DataT]:
        pass

    @abstractmethod
    def get_data(self, *args: Any, **kwargs: Any) -> DataT | None:
        pass

    @abstractmethod
    def get_datas(self) -> list[DataT]:
        pass

    @abstractmethod
    def update_df(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def data_as_features(self) -> bool:
        return self._data_as_features is True

    def set_data_as_features(self, as_features: bool) -> None:
        if self._data_as_features is not None and as_features != self._data_as_features:
            raise ValueError(
                f"{self.__class__.__name__} already has as_features="
                + f"{self._data_as_features}; it cannot be changed to {as_features}"
            )
        self._data_as_features = as_features

    def _create_feed(self, data: DataT) -> FeedT:
        from pfeed.feeds import create_feed

        return create_feed(  # pyright: ignore[reportReturnType]
            data_source=data.source,
            data_category=data.category,
            pipeline_mode=True,
            num_workers=data.config.num_batch_workers,
        )

    @staticmethod
    def _create_cache_storage_config(storage_config: StorageConfig) -> StorageConfig:
        """Create a cache storage config inheriting io settings from the original storage config."""
        return storage_config.model_copy(
            update={
                "storage": DataStorage.CACHE,
                "data_layer": DataLayer.CURATED,
            }
        )

    def _standardize_df(self, df: IntoDataFrame) -> nw.DataFrame[Any]:
        """Adds metadata columns to the dataframe"""
        from pfund.enums import SourceType

        metadata = {
            "source_type": nw.lit(SourceType.BATCH).cast(nw.String),
        }
        assert set(metadata) == set(self.METADATA_COLS), (
            f"metadata keys {set(metadata)} do not match METADATA_COLS {set(self.METADATA_COLS)}"
        )
        nwdf = nw.from_native(df).with_columns(
            # product=nw.lit(data.product.name).cast(nw.String),
            **metadata,
        )
        cols = nwdf.collect_schema().names()
        # re-order columns
        target_cols = self.LEFT_COLS + [
            col for col in cols if col not in self.LEFT_COLS
        ]
        nwdf = nwdf.select(target_cols)
        return nwdf

    def pivot_df(self, df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        """Pivots data dataframe from long form to wide form.

        Args:
            df: data_df in long form
        """
        # return df.pivot(
        #     on=self.PIVOT_COLS,
        #     index=self.INDEX_COL,
        # ).sort(self.INDEX_COL)
        return pivot_long_to_wide(
            df,
            index_col=self.INDEX_COL,
            pivot_cols=self.PIVOT_COLS,
            key_cols=self.KEY_COLS,
        )

    def get_df(
        self,
        window_size: int | None = None,
        pivot: bool = False,
        to_native: bool = False,
    ) -> nw.DataFrame[Any] | IntoDataFrame:
        if self._df is None:
            raise RuntimeError(f"{self.__class__.__name__} data df is not ready")
        df = self._df if window_size is None else self._df.tail(window_size)
        if pivot:
            df = self.pivot_df(df)
        return df.to_native() if to_native else df
