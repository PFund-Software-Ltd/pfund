from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generic, TypeVar, ClassVar

if TYPE_CHECKING:
    from narwhals._native import NativeDataFrame
    from pfund.datas.databoy import DataBoy

from abc import ABC, abstractmethod

import narwhals as nw

from pfeed.feeds.base_feed import BaseFeed
from pfeed.enums import DataStorage, DataLayer
from pfeed.storages.storage_config import StorageConfig
from pfund.datas.data_base import BaseData


DataT = TypeVar('DataT', bound=BaseData)
FeedT = TypeVar('FeedT', bound=BaseFeed)


class BaseDataStore(ABC, Generic[DataT, FeedT]):
    LEFT_COLS: ClassVar[list[str]]
    INDEX_COL: ClassVar[str] = 'date'
    PIVOT_COLS: ClassVar[list[str]]
    METADATA_COLS: ClassVar[list[str]] = ['source_type']

    def __init__(self, databoy: DataBoy):
        self._databoy: DataBoy = databoy
        self._logger = databoy.logger
        self._df: nw.DataFrame[Any] | None = None
    
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

    def _create_feed(self, data: DataT) -> FeedT:
        from pfeed.feeds import create_feed
        return create_feed(  # pyright: ignore[reportReturnType]
            data_source=data.source,
            data_category=data.category,
            pipeline_mode=True,
            num_workers=data.config.num_batch_workers,
        )
    
    def _create_cache_storage_config(self, storage_config: StorageConfig) -> StorageConfig:
        '''Create a cache storage config inheriting io settings from the original storage config.'''
        return StorageConfig(
            storage=DataStorage.CACHE,
            data_path=storage_config.data_path,
            data_layer=DataLayer.CURATED,
            io_format=storage_config.io_format,
            compression=storage_config.compression,
        )
    
    def _standardize_df(self, df: NativeDataFrame) -> nw.DataFrame[Any]:
        '''Adds metadata columns to the dataframe'''
        from pfund.enums import SourceType
        metadata = {
            'source_type': nw.lit(SourceType.BATCH).cast(nw.String),
        }
        assert set(metadata) == set(self.METADATA_COLS), (
            f'metadata keys {set(metadata)} do not match METADATA_COLS {set(self.METADATA_COLS)}'
        )
        nwdf = (
            nw
            .from_native(df)
            .with_columns(
                # product=nw.lit(data.product.name).cast(nw.String),
                **metadata,
            )
        )
        cols = nwdf.collect_schema().names()
        # re-order columns
        target_cols = self.LEFT_COLS + [col for col in cols if col not in self.LEFT_COLS]
        nwdf = nwdf.select(target_cols)
        return nwdf

    def pivot_df(self, df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        '''Pivots data dataframe from long form to wide form.
        Args:
            df: data_df in long form
        '''
        return (
            df
            .pivot(
                on=self.PIVOT_COLS,
                index=self.INDEX_COL,
            )
            .sort(self.INDEX_COL)
        )
    
    def get_df(self, window_size: int | None = None, pivot: bool = False, to_native: bool = False) -> nw.DataFrame[Any] | NativeDataFrame | None:
        if self._df is None:
            return None
        df = self._df if window_size is None else self._df.tail(window_size)
        if pivot:
            df = self.pivot_df(df)
        return df.to_native() if to_native else df
