from __future__ import annotations
from collections import defaultdict

from typing import TYPE_CHECKING, Generator
if TYPE_CHECKING:
    from pfund.datas.data_base import BaseData
    
import pandas as pd
import polars as pl

from pfund.data_tools.data_tool_base import BaseDataTool
from pfund.utils.envs import backtest


class PolarsDataTool(BaseDataTool):
    _INDEX = ['ts', 'product', 'resolution']
    
    def get_df(self, copy=True):
        return self.df.clone() if copy else self.df
    
    def concat(self, dfs: list[pl.DataFrame | pl.LazyFrame]) -> pl.DataFrame | pl.LazyFrame:
        return pl.concat(dfs)
    
    def prepare_df(self):
        assert self._raw_dfs, "No data is found, make sure add_data(...) is called correctly"
        self.df = pl.concat(self._raw_dfs.values())
        self.df = self.df.sort(by='ts', descending=False)
        # arrange columns
        self.df = self.df.select(self._INDEX + [col for col in self.df.columns if col not in self._INDEX])
        self._raw_dfs.clear()
    
    def get_total_rows(self, df: pl.DataFrame | pl.LazyFrame):
        if isinstance(df, pl.DataFrame):
            return df.shape[0]
        elif isinstance(df, pl.LazyFrame):
            return df.count().collect()['ts'][0]
        else:
            raise ValueError("df should be either pl.DataFrame or pl.LazyFrame")
        
    @backtest
    def iterate_df_by_chunks(self, lf: pl.LazyFrame, num_chunks=1) -> Generator[pd.DataFrame, None, None]:
        total_rows = self.get_total_rows(lf)
        chunk_size = total_rows // num_chunks
        for i in range(0, total_rows, chunk_size):
            df_chunk = lf.slice(i, chunk_size).collect()
            yield df_chunk
    
    @backtest
    def preprocess_event_driven_df(self, df: pl.DataFrame) -> pl.DataFrame:
        def _check_resolution(res):
            from pfund.datas.resolution import Resolution
            resolution = Resolution(res)
            return {
                'is_quote': resolution.is_quote(),
                'is_tick': resolution.is_tick()
            }
    
        df = df.with_columns(
            # converts 'ts' from datetime to unix timestamp
            pl.col("ts").cast(pl.Int64) // 10**6 / 10**3,
            
            # add 'broker', 'is_quote', 'is_tick' columns
            pl.col('product').str.split("-").list.get(0).alias("broker"),
            pl.col('resolution').map_elements(
                _check_resolution,
                return_dtype=pl.Struct([
                    pl.Field('is_quote', pl.Boolean), 
                    pl.Field('is_tick', pl.Boolean)
                ])
            ).alias('Resolution')
        ).unnest('Resolution')
        
        # arrange columns
        left_cols = self._INDEX + ['broker', 'is_quote', 'is_tick']
        df = df.select(left_cols + [col for col in df.columns if col not in left_cols])
        return df
    
    @backtest
    def postprocess_vectorized_df(self, df: pl.DataFrame) -> pl.LazyFrame:
        return df.lazy()
    
    # TODO:
    def prepare_df_with_signals(self, models):
        pass
    
    # TODO: for train engine
    def prepare_datasets(self, datas):
        pass
    
    def clear_df(self):
        self.df.clear()
    
    # TODO:
    def append_to_df(self, data: BaseData, predictions: dict, **kwargs):
        pass

    
    '''
    ************************************************
    Helper Functions
    ************************************************
    '''
    def output_df_to_parquet(self, df: pl.DataFrame | pl.LazyFrame, file_path: str, compression: str='zstd'):
        df.write_parquet(file_path, compression=compression)
    
    # TODO
    def filter_df(self, df: pl.DataFrame | pl.LazyFrame, **kwargs) -> pl.DataFrame | pl.LazyFrame:
        pass
    
    # TODO
    def unstack_df(self, df: pl.DataFrame | pl.LazyFrame, **kwargs) -> pl.DataFrame | pl.LazyFrame:
        pass