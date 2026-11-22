from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import polars as pl
    from pfeed.io.parquet_io import ParquetIO

    from pfund.data_config import DataConfig
    from pfund.databoy import DataBoy
    from pfund.typing import Component

from abc import ABC, abstractmethod

from pfeed.base.feed import BaseFeed
from pfund_blocks.datas.data_base import BaseData


class BaseDataStore[DataT: BaseData, FeedT: BaseFeed](ABC):
    METADATA_COLS: ClassVar[list[str]] = []

    def __init__(self, databoy: DataBoy):
        self._databoy: DataBoy = databoy
        self._logger = databoy._logger
        self._df: pl.DataFrame | None = None  # in long form
        self._data_as_features: bool | None = None
        # keyed by the data itself, see BaseData.key in pfund_blocks
        self._configs: dict[DataT, DataConfig] = {}

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

    def get_config(self, data: DataT) -> DataConfig:
        return self._configs[data]

    def _set_data_as_features(self, as_features: bool) -> None:
        if self._data_as_features is None:
            self._data_as_features = as_features
        elif as_features != self._data_as_features:
            # as_features applies to the whole store, i.e. all data of this category (e.g. all market data),
            # it can't differ per call. other categories (e.g. news data) have their own stores and flags.
            raise ValueError(
                f"{self.__class__.__name__} has already set data_as_features={self._data_as_features}, "
                + f"got inconsistent {as_features=}"
            )

    def _create_io_for_caching(self) -> ParquetIO:
        """Create a cache storage config inheriting io settings from the original storage config."""
        import pfeed as pe

        engine_context = self._component.context
        return pe.ParquetIO(base_path=engine_context.config.cache_path)

    def _standardize_df(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Adds metadata columns to the dataframe"""
        import polars as pl

        from pfund.enums import SourceType

        metadata = {
            "source_type": pl.lit(SourceType.BATCH.value, dtype=pl.String),
        }
        assert set(metadata) == set(self.METADATA_COLS), (
            f"metadata keys {set(metadata)} do not match METADATA_COLS {set(self.METADATA_COLS)}"
        )
        return df.with_columns(
            # product=pl.lit(data.product.name, dtype=pl.String),
            **metadata,
        )

    def get_df(self, window_size: int | None = None) -> pl.DataFrame:
        if self._df is None:
            raise RuntimeError(f"{self.__class__.__name__} data df is not ready")
        return self._df if window_size is None else self._df.tail(window_size)
