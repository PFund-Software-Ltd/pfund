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
        if self._data_as_features is None:
            self._data_as_features = as_features
        elif as_features != self._data_as_features:
            self._logger.debug(
                f"{self.__class__.__name__} has already set data_as_features={self._data_as_features}, "
                + f"ignoring inconsistent input {as_features=}"
            )

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
        from pfund.utils.dataframe import pivot_long_to_wide

        return pivot_long_to_wide(
            df,
            index_col=self.INDEX_COL,
            pivot_cols=self.PIVOT_COLS,
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
