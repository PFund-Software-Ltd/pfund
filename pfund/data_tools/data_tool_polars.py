from __future__ import annotations
from collections import defaultdict
from decimal import Decimal

from typing import TYPE_CHECKING, Iterator
if TYPE_CHECKING:
    from pfund.datas.data_base import BaseData
    
import polars as pl

from pfund.data_tools.data_tool_base import BaseDataTool
from pfund.utils.envs import backtest


class PolarsDataTool(BaseDataTool):
    _INDEX = ['ts', 'product', 'resolution']
    
    def prepare_df(self):
        assert self._raw_dfs, "No data is found, make sure add_data(...) is called correctly"
        self.df = pl.concat(self._raw_dfs.values())
        self.df = self.df.sort(by='ts', descending=False)
        # arrange columns
        self.df = self.df.select(self._INDEX + [col for col in self.df.columns if col not in self._INDEX])
        self._raw_dfs.clear()
    
    @backtest
    def preprocess_event_driven_df(self, df: pl.DataFrame | pl.LazyFrame) -> Iterator:
        pass
    
    @backtest
    def postprocess_vectorized_df(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        pass
    
    # TODO:
    def prepare_df_with_signals(self, models):
        pass
    
    # TODO: for train engine
    def prepare_datasets(self, datas):
        pass
    
    # TODO:
    def clear_df(self):
        pass
    
    # TODO:
    def append_to_df(self, data: BaseData, predictions: dict, **kwargs):
        pass

    def output_df_to_parquet(self, df: pl.DataFrame | pl.LazyFrame, file_path: str):
        df.write_parquet(file_path, compression='zstd')