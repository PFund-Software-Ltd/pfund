from collections import defaultdict
from decimal import Decimal

import pandas as pd

from pfund.products.product_base import BaseProduct
from pfund.datas.data_base import BaseData
from pfund.datas.resolution import Resolution
from pfund.data_tools.data_tool_base import BaseDataTool
from pfund.utils.envs import backtest
from pfund.utils.utils import get_engine_class


Engine = get_engine_class()


class PandasDataTool(BaseDataTool):
    '''Data Tool for backtesting using pandas'''
    _INDEX = ['ts', 'product', 'resolution']
    _GROUP = ['product', 'resolution']
    _DECIMAL_COLS = ['price', 'open', 'high', 'low', 'close', 'volume']
    
    @backtest
    def prepare_df_after_vectorized_backtesting(self):
        '''Prepares the df after vectorized backtesting, including:
        if has 'orders' column:
        - converts orders to trades
        if no 'orders' column, then 'trade_size'/'position' must be in the df
        - derives 'trade_size'/'position' from 'position'/'trade_size'
        '''
        cols = self.df.columns
        if 'position' not in cols and 'trade_size' not in cols:
            raise Exception("either 'position' or 'trade_size' must be in the dataframe columns")
        # use 'position' to derive 'trade_size' and vice versa
        elif 'position' in cols and 'trade_size' not in cols:
            self.df['trade_size'] = self.df['position'].diff(1)
            # fill the nan value in the first row with the initial position
            self.df.iloc[0, self.df.columns.get_loc('trade_size')] = self.df.iloc[0, self.df.columns.get_loc('position')]
        elif 'trade_size' in cols and 'position' not in cols:
            self.df['position'] = self.df['trade_size'].cumsum()
        
        if 'trade_price' not in cols:
            self.df.loc[self.df['trade_size'] != 0, 'trade_price'] = self.df['close']

    @backtest
    def prepare_df_before_event_driven_backtesting(self):
        self.df.reset_index(inplace=True)
        # converts 'ts' from datetime to unix timestamp
        self.df['ts'] = self.df['ts'].astype(int) // 10**6  # in milliseconds
        self.df['ts'] = self.df['ts'] / 10**3  # in seconds with milliseconds precision
        # convert float columns to decimal for consistency with live trading
        for col in self.df.columns:
            if col in self._DECIMAL_COLS:
                self.df[col] = self.df[col].apply(lambda x: Decimal(str(x)))
        # TODO: split 'broker' str column from 'product' str column
        # self.df['broker'] = ...
    
    def prepare_df(self):
        assert self._raw_dfs, "No data is found, make sure add_data(...) is called correctly"
        self.df = pd.concat(self._raw_dfs.values())
        # sort first, so that product column is in order
        self.df.set_index(self._INDEX, inplace=True)
        self.df.sort_index(level='ts', inplace=True)
        self._raw_dfs.clear()

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
    
    def _prepare_datasets(self, datas):
        # create datasets based on train/val/test periods
        datasets = defaultdict(list)  # {'train': [df_of_product_1, df_of_product_2]}
        for product in datas:
            for type_, periods in [('train', self.train_periods), ('val', self.val_periods), ('test', self.test_periods)]:
                period = periods[product]
                if period is None:
                    raise Exception(f'{type_}_period for {product} is not specified')
                df = self.get_df(self.df, start_date=period[0], end_date=period[1], symbol=product.symbol).reset_index()
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
    
    def get_df_iterable(self) -> list:
        return self.df.itertuples(index=False)
        
    def get_df_unstacked(self, df: pd.DataFrame | None=None):
        df = self.df if df is None else df
        return df.unstack(level=self._GROUP)
    
    def get_df_ffill(self, df: pd.DataFrame | None=None):
        df = self.df if df is None else df
        return (
            df
            .unstack(level=self._GROUP)
            .ffill()
            .stack(level=self._GROUP)
        )
    
    def get_df(self, df: pd.DataFrame | None=None, start_date: str | None=None, end_date: str | None=None, product: BaseProduct | None=None, resolution: str=''):
        df = self.df if df is None else df
        assert df is not None, "df is None, make sure strategy.start()/model.start() is called."
        product = product or slice(None)
        if resolution:
            if Engine.mode == 'event_driven':
                resolution = Resolution(resolution)
            else:
                resolution = resolution
        else:
            resolution = slice(None)
        return df.loc[(slice(start_date, end_date), product, resolution), :]
    
    def get_df_products(self, df: pd.DataFrame | None=None) -> list:
        df = self.df if df is None else df
        return df.index.get_level_values('product').unique().to_list()
    
    def get_df_resolutions(self, df: pd.DataFrame | None=None) -> list:
        df = self.df if df is None else df
        return df.index.get_level_values('resolution').unique().to_list()

    def convert_ts_index_to_dt(self, df: pd.DataFrame) -> pd.DataFrame:
        ts_index = df.index.get_level_values('ts')
        dt_index = pd.to_datetime(ts_index, unit='s')
        df.index = df.index.set_levels(dt_index, level='ts')
        return df
    
    def clear_df(self):
        index_names = self.df.index.names
        self.df = pd.DataFrame(
            columns=self.df.columns,
            index=pd.MultiIndex(levels=[[]]*len(index_names), codes=[[]]*len(index_names), names=index_names)
        )
    
    def output_df_to_parquet(self, df: pd.DataFrame, file_path: str):
        df.to_parquet(file_path)
    
    def create_multi_index(self, index_data: dict, index_names: list[str]) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples([tuple(index_data[name] for name in index_names)], names=index_names)
        
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
    
    def rescale_df(
        self, 
        window_size: int | None=None,
        min_periods: int=20,
        df: pd.DataFrame | None=None
    ) -> pd.DataFrame:
        """Scales the data to z-score using a rolling window to avoid lookahead bias
        If window_size is None, then use expanding window
        """
        df = self.df if df is None else df
        if window_size:
            mu = df.rolling(window=window_size, min_periods=min_periods).mean()
            sigma = df.rolling(window=window_size, min_periods=min_periods).std()
        else:
            mu = df.expanding(min_periods=min_periods).mean()
            sigma = df.expanding(min_periods=min_periods).std()
        df_norm = (df - mu) / sigma
        return df_norm
    