from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from pfund.databoy import DataBoy
    from pfund.typing import Component

from abc import ABC, abstractmethod

import narwhals as nw
from pfeed.base.feed import BaseFeed
from pfeed.enums import DataLayer
from pfund_blocks.datas.data_base import BaseData


class BaseDataStore[DataT: BaseData, FeedT: BaseFeed](ABC):
    METADATA_COLS: ClassVar[list[str]] = []

    def __init__(self, databoy: DataBoy):
        self._databoy: DataBoy = databoy
        self._logger = databoy._logger
        self._df: nw.DataFrame[Any] | None = None  # in long form
        self._data_as_features: bool | None = None

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
    def _component(self) -> Component:
        return self._databoy._component

    @property
    def data_as_features(self) -> bool:
        return self._data_as_features is True

    def _set_data_as_features(self, as_features: bool) -> None:
        if self._data_as_features is None:
            self._data_as_features = as_features
        elif as_features != self._data_as_features:
            self._logger.debug(
                f"{self.__class__.__name__} has already set data_as_features={self._data_as_features}, "
                + f"ignoring inconsistent input {as_features=}"
            )

    def _create_feed(self, data: DataT) -> FeedT:
        import pfeed as pe

        return pe.get_feed(
            data.source,
            data.category,
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
        return nw.from_native(df).with_columns(
            # product=nw.lit(data.product.name).cast(nw.String),
            **metadata,
        )

    def get_df(
        self,
        window_size: int | None = None,
        to_native: bool = False,
    ) -> nw.DataFrame[Any] | IntoDataFrame:
        if self._df is None:
            raise RuntimeError(f"{self.__class__.__name__} data df is not ready")
        df = self._df if window_size is None else self._df.tail(window_size)
        return df.to_native() if to_native else df
