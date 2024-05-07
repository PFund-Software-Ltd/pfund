from __future__ import annotations
from collections import defaultdict

from typing import TYPE_CHECKING, Generator, Literal
if TYPE_CHECKING:
    from pfund.datas.data_base import BaseData
    from pfund.models.model_base import BaseModel

import numpy as np    
import polars as pl

from pfund.strategies.strategy_base import BaseStrategy
from pfund.data_tools.data_tool_base import BaseDataTool
from pfund.utils.envs import backtest


class PolarsDataTool(BaseDataTool):
    # TODO:
    def get_df(self, copy=True) -> pl.LazyFrame:
        return self.df.clone() if copy else self.df
    
    def prepare_df(self, ts_col_type: Literal['datetime', 'timestamp']='datetime'):
        assert self._raw_dfs, "No data is found, make sure add_data(...) is called correctly"
        self.df: pl.LazyFrame = pl.concat(self._raw_dfs.values())
        self.df = self.df.sort(by=self.INDEX, descending=False)
        # arrange columns
        self.df = self.df.select(self.INDEX + [col for col in self.df.columns if col not in self.INDEX])
        if ts_col_type == 'datetime':
            # ts column already is datetime by default
            pass
        elif ts_col_type == 'timestamp':
            # converts 'ts' from datetime to unix timestamp
            self.df = self.df.with_columns(
                pl.col("ts").cast(pl.Int64) // 10**6 / 10**3,
            )
        self._raw_dfs.clear()
    
    def clear_df(self):
        self.df.clear()
    
    # TODO:
    def _push_new_rows_to_df(self):
        pass
    
    # TODO:
    def append_to_df(self, data: BaseData, predictions: dict, **kwargs):
        pass

    # TODO
    def signalize(self, X: pl.LazyFrame, pred_y: np.ndarray, columns: list[str]) -> pl.LazyFrame:
        pass

    @staticmethod
    @backtest
    def iterate_df_by_chunks(lf: pl.LazyFrame, num_chunks=1) -> Generator[pl.LazyFrame, None, None]:
        total_rows = lf.count().collect()['ts'][0]
        chunk_size = total_rows // num_chunks
        for i in range(0, total_rows, chunk_size):
            df_chunk = lf.slice(i, chunk_size)
            yield df_chunk
    
    @backtest
    def preprocess_event_driven_df(self, df: pl.LazyFrame) -> pl.LazyFrame:
        def _check_resolution(res):
            from pfund.datas.resolution import Resolution
            resolution = Resolution(res)
            return {
                'is_quote': resolution.is_quote(),
                'is_tick': resolution.is_tick()
            }
    
        df = df.with_columns(
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
        left_cols = self.INDEX + ['broker', 'is_quote', 'is_tick']
        df = df.select(left_cols + [col for col in df.columns if col not in left_cols])
        return df
    
    # TODO
    @backtest
    def preprocess_vectorized_df(self, df: pl.LazyFrame, backtestee: BaseStrategy | BaseModel) -> pl.LazyFrame:
        pass
    
    # TODO
    @staticmethod
    @backtest
    def postprocess_vectorized_df(df_chunks: list[pl.LazyFrame]) -> pl.LazyFrame:
        df = pl.concat(df_chunks)
        return df
    
    # TODO: for train engine
    def prepare_datasets(self, datas):
        pass
    
    
    '''
    ************************************************
    Helper Functions
    ************************************************
    '''
    @staticmethod
    def output_df_to_parquet(df: pl.DataFrame | pl.LazyFrame, file_path: str, compression: str='zstd'):
        df.write_parquet(file_path, compression=compression)
    