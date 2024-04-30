from __future__ import annotations
from collections import defaultdict

from typing import TYPE_CHECKING, Generator
if TYPE_CHECKING:
    from pfund.datas.data_base import BaseData

import pandas as pd

from pfund.data_tools.data_tool_base import BaseDataTool
from pfund.utils.envs import backtest


class PandasDataTool(BaseDataTool):
    _INDEX = ['ts', 'product', 'resolution']
    _GROUP = ['product', 'resolution']
    
    def get_df(self, copy=True):
        return self.df.copy(deep=True) if copy else self.df
    
    def concat(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(dfs)
    
    def prepare_df(self):
        assert self._raw_dfs, "No data is found, make sure add_data(...) is called correctly"
        self.df = pd.concat(self._raw_dfs.values())
        self.df = self.df.sort_values(by='ts', ascending=True)
        # arrange columns
        self.df = self.df[self._INDEX + [col for col in self.df.columns if col not in self._INDEX]]
        self._raw_dfs.clear()
    
    @backtest
    def iterate_df_by_chunks(self, df: pd.DataFrame, num_chunks=1) -> Generator[pd.DataFrame, None, None]:
        total_rows = df.shape[0]
        chunk_size = total_rows // num_chunks
        for i in range(0, total_rows, chunk_size):
            df_chunk = df.iloc[i:i + chunk_size].copy(deep=True)
            yield df_chunk
    
    @backtest
    def preprocess_event_driven_df(self, df: pd.DataFrame) -> pd.DataFrame:
        def _check_resolution(res):
            from pfund.datas.resolution import Resolution
            resolution = Resolution(res)
            return resolution.is_quote(), resolution.is_tick()
        
        # converts 'ts' from datetime to unix timestamp
        # in milliseconds int -> in seconds with milliseconds precision
        df['ts'] = df['ts'].astype(int) // 10**6 / 10**3
        
        # add 'broker', 'is_quote', 'is_tick' columns
        df['broker'] = df['product'].str.split('-').str[0]
        df['is_quote'], df['is_tick'] = zip(*df['resolution'].apply(_check_resolution))
        
        # arrange columns
        left_cols = self._INDEX + ['broker', 'is_quote', 'is_tick']
        df = df[left_cols + [col for col in df.columns if col not in left_cols]]
        return df
    
    @backtest
    def postprocess_vectorized_df(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Prepares the df after vectorized backtesting, including:
        if has 'orders' column:
        - converts orders to trades
        if no 'orders' column, then 'trade_size'/'position' must be in the df
        - derives 'trade_size'/'position' from 'position'/'trade_size'
        '''
        cols = df.columns
        if 'position' not in cols and 'trade_size' not in cols:
            raise Exception("either 'position' or 'trade_size' must be in the dataframe columns")
        # use 'position' to derive 'trade_size' and vice versa
        elif 'position' in cols and 'trade_size' not in cols:
            df['trade_size'] = df['position'].diff(1)
            # fill the nan value in the first row with the initial position
            df.iloc[0, df.columns.get_loc('trade_size')] = df.iloc[0, df.columns.get_loc('position')]
        elif 'trade_size' in cols and 'position' not in cols:
            df['position'] = df['trade_size'].cumsum()
        
        if 'trade_price' not in cols:
            df.loc[df['trade_size'] != 0, 'trade_price'] = df['close']
        return df
    
    def prepare_df_with_signals(self, models):
        # NOTE: models can have different ts_ranges, need to store the original ts_range before concatenating
        ts_range = self.df.index.get_level_values('ts')
        for mdl, model in models.items():
            assert model.signal is not None, f"signal is None, please make sure model '{mdl}' is loaded or was dumped using 'model.dump(signal)' correctly."
            # rename model columns to avoid conflict
            num_model_cols = len(model.signal.columns)
            new_model_cols = {col: model.name if num_model_cols == 1 else model.name+'_'+col for col in model.signal.columns}
            model.signal = model.signal.rename(columns=new_model_cols)
            # filter to match the timestamp range
            model.signal = model.signal[ts_range.min():ts_range.max()]
            self.df = pd.concat([self.df, model.signal], axis=1)
        self.df.sort_index(level='ts', inplace=True)
    
    # TODO: for train engine
    def prepare_datasets(self, datas):
        # create datasets based on train/val/test periods
        datasets = defaultdict(list)  # {'train': [df_of_product_1, df_of_product_2]}
        for product in datas:
            for type_, periods in [('train', self.train_periods), ('val', self.val_periods), ('test', self.test_periods)]:
                period = periods[product]
                if period is None:
                    raise Exception(f'{type_}_period for {product} is not specified')
                df = self.filter_df(self.df, start_date=period[0], end_date=period[1], symbol=product.symbol).reset_index()
                datasets[type_].append(df)
                
        # combine datasets from different products to create the final train/val/test set
        for type_ in ['train', 'val', 'test']:
            df = pd.concat(datasets[type_])
            df.set_index(self._INDEX, inplace=True)
            df.sort_index(level='ts', inplace=True)
            if type_ == 'train':
                self.train_set = df
            elif type_ == 'val':
                self.val_set = self.validation_set = df
            elif type_ == 'test':
                self.test_set = df
    
    def clear_df(self):
        index_len = len(self._INDEX)
        self.df = pd.DataFrame(
            columns=self.df.columns,
            index=pd.MultiIndex(levels=[[]]*index_len, codes=[[]]*index_len, names=self._INDEX)
        )
    
    # OPTIMIZE
    def append_to_df(self, data: BaseData, predictions: dict, **kwargs):
        '''Appends new data to the df
        The flow is, the df is cleared in model's event-driven backtesting,
        data & prediction (single signal) will be gradually appended back to the df for model.next() to use.
        '''
        row_data = {}
        for col in self.df.columns:
            if hasattr(data, col):
                row_data[col] = getattr(data, col)
            elif col in kwargs:
                row_data[col] = kwargs[col]
        for mdl, pred_y in predictions.items():
            if pred_y is not None and pred_y.shape[0] == 1:
                row_data[mdl] = pred_y[0]
            else:
                row_data[mdl] = pred_y
        index_data = {'ts': data.dt, 'product': repr(data.product), 'resolution': repr(data.resolution)}
        new_row = pd.DataFrame(
            [row_data], 
            index=self.create_multi_index(index_data, self.df.index.names)
        )
        self.df = pd.concat([self.df, new_row], ignore_index=False)
    
    def create_multi_index(self, index_data: dict, index_names: list[str]) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples([tuple(index_data[name] for name in index_names)], names=index_names)
       
       
    '''
    ************************************************
    Helper Functions
    ************************************************
    '''
    def get_index_values(self, df: pd.DataFrame, index: str) -> list:
        assert index in df.index.names, f"index must be one of {df.index.names}"
        return df.index.get_level_values(index).unique().to_list()
    
    def set_index_values(self, df: pd.DataFrame, index: str, values: list) -> pd.DataFrame:
        assert index in df.index.names, f"index must be one of {df.index.names}"
        df.index = df.index.set_levels(values, level=index)
        return df
    
    def output_df_to_parquet(self, df: pd.DataFrame, file_path: str, compression: str='zstd'):
        df.to_parquet(file_path, compression=compression)
    
    def filter_df(self, df: pd.DataFrame, start_date: str | None=None, end_date: str | None=None, product: str='', resolution: str=''):
        product = product or slice(None)
        resolution = resolution or slice(None)
        return df.loc[(slice(start_date, end_date), product, resolution), :]
    
    def unstack_df(self, df: pd.DataFrame):
        return df.unstack(level=self._GROUP)
    
    def ffill_df(self, df: pd.DataFrame):
        return (
            self.unstack_df(df)
            .ffill()
            .stack(level=self._GROUP)
        )
        
    def rescale_df(
        self, 
        df: pd.DataFrame,
        window_size: int | None=None,
        min_periods: int=20,
    ) -> pd.DataFrame:
        """Scales the data to z-score using a rolling window to avoid lookahead bias
        If window_size is None, then use expanding window
        """
        if window_size:
            mu = df.rolling(window=window_size, min_periods=min_periods).mean()
            sigma = df.rolling(window=window_size, min_periods=min_periods).std()
        else:
            mu = df.expanding(min_periods=min_periods).mean()
            sigma = df.expanding(min_periods=min_periods).std()
        df_norm = (df - mu) / sigma
        return df_norm
    