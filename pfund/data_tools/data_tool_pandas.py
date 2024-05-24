from __future__ import annotations
from typing import TYPE_CHECKING, Generator, Literal
if TYPE_CHECKING:
    from pfund.datas.data_base import BaseData

from collections import defaultdict

import numpy as np
import pandas as pd
from rich.console import Console

from pfund.data_tools.data_tool_base import BaseDataTool
from pfund.utils.envs import backtest, train


SUPPORTED_CLOSE_POSITION_WHEN = ['signal_flip', 'no_signal']
tSUPPORTED_CLOSE_POSITION_WHEN = Literal['signal_flip', 'no_signal']


# NOTE: convention: all function names that endswith "_df" will directly modify self.df, e.g. "xxx_df"
class PandasDataTool(BaseDataTool):
    def get_df(
        self, 
        start_idx: int=0, 
        end_idx: int | None=None, 
        product: str | None=None, 
        resolution: str | None=None, 
        copy: bool=True
    ) -> pd.DataFrame | None:
        if self.df is None:
            return
        if self._new_rows:
            self._push_new_rows_to_df()
        df = self.df.copy(deep=True) if copy else self.df
        if product and resolution:
            df = df.loc[(df['product'] == product) & (df['resolution'] == resolution)]
        elif product and not resolution:
            df = df.loc[df['product'] == product]
        elif not product and resolution:
            df = df.loc[df['resolution'] == resolution]
        return df.iloc[start_idx:end_idx]
    
    def prepare_df(self, ts_col_type: Literal['datetime', 'timestamp']='datetime'):
        assert self._raw_dfs, "No data is found, make sure add_data(...) is called correctly"
        self.df = pd.concat(self._raw_dfs.values())
        self.df.sort_values(by=self.INDEX, ascending=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        # arrange columns
        self.df = self.df[self.INDEX + [col for col in self.df.columns if col not in self.INDEX]]
        if ts_col_type == 'datetime':
            # ts column already is datetime by default
            pass
        elif ts_col_type == 'timestamp':
            # converts 'ts' from datetime to unix timestamp
            # in milliseconds int -> in seconds with milliseconds precision
            self.df['ts'] = self.df['ts'].astype(int) // 10**6 / 10**3
        self._raw_dfs.clear()

    def clear_df(self):
        self.df = pd.DataFrame(columns=self.df.columns).astype(self.df.dtypes)
        
    def append_to_df(self, data: BaseData, predictions: dict, **kwargs):
        '''Appends new data to the df
        Args:
            kwargs: other_info about the data
        '''
        row_data = {
            'ts': data.ts, 
            'product': repr(data.product), 
            'resolution': data.resol
        }
        
        for col in self.df.columns:
            if col in self.INDEX:
                continue
            # e.g. open, high, low, close, volume
            if hasattr(data, col):
                row_data[col] = getattr(data, col)
            elif col in kwargs:
                row_data[col] = kwargs[col]
                
        for mdl, pred_y in predictions.items():
            row_data[mdl] = pred_y
        
        self._new_rows.append(row_data)
        if len(self._new_rows) >= self._MAX_NEW_ROWS:
            self._push_new_rows_to_df()
    
    def _push_new_rows_to_df(self):
        new_rows_df = pd.DataFrame(self._new_rows)
        self.df = pd.concat([self.df, new_rows_df], ignore_index=True)
        self.df.sort_values(by=self.INDEX, ascending=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self._new_rows.clear()
        if self._MAX_ROWS and self.df.shape[0] >= self._MAX_ROWS:
            self._trim_df()
            
    def _trim_df(self):
        num_rows_to_trim = self.df.shape[0] - self._MIN_ROWS
        trimmed_df = self.df.iloc[:num_rows_to_trim]
        # TODO:
        self._write_df_to_db(trimmed_df)
        self.df = self.df.iloc[-self._MIN_ROWS:]
    
    # TODO:
    def _write_df_to_db(self, trimmed_df: pd.DataFrame):
        pass
    
    @staticmethod
    def get_nan_columns(df: pd.DataFrame) -> list[str]:
        all_nan_columns: pd.Series = df.isna().all()
        nan_columns = all_nan_columns[all_nan_columns].index.tolist()
        return nan_columns
    
    @staticmethod
    def assert_frame_equal(df1: pd.DataFrame, df2: pd.DataFrame):
        '''
        Raise:
            AssertionError: if df1 and df2 are not equal
        '''
        pd.testing.assert_frame_equal(df1, df2, check_exact=False, rtol=1e-5)
    
    @backtest
    def merge_signal_dfs_with_df(self, signal_dfs: list[pd.DataFrame]):
        for signal_df in signal_dfs:
            self.df = pd.merge(self.df, signal_df, on=self.INDEX, how='left')
        self.df.sort_values(by=self.INDEX, ascending=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
    @backtest
    def signalize(self, X: pd.DataFrame, pred_y: np.ndarray, columns: list[str]) -> pd.DataFrame:
        pred_df = pd.DataFrame(pred_y, columns=columns)
        assert set(self.INDEX) <= set(X.columns), f"{self.INDEX} must be in X's columns"
        X = X[self.INDEX]  # only keep the index columns
        signal_df = pd.concat([X, pred_df], axis=1)
        # arrange columns
        signal_df = signal_df[self.INDEX + [col for col in signal_df.columns if col not in self.INDEX]]
        return signal_df

    @staticmethod
    @backtest
    def iterate_df_by_chunks(df: pd.DataFrame, num_chunks=1) -> Generator[pd.DataFrame, None, None]:
        total_rows = df.shape[0]
        chunk_size = total_rows // num_chunks
        for i in range(0, total_rows, chunk_size):
            df_chunk = df.iloc[i:i + chunk_size].copy(deep=True)
            yield df_chunk

    @backtest
    def _create_signal(
        self,
        df: pd.DataFrame,
        buy_condition: pd.Series | None=None,
        sell_condition: pd.Series | None=None,
        first_only: bool=False,
    ) -> pd.DataFrame:
        if buy_condition is None and sell_condition is None:
            raise ValueError("Either buy or sell must be provided")
        elif buy_condition is not None and sell_condition is not None:
            # assert non-overlapping signals
            overlaps = buy_condition & sell_condition
            if overlaps.any():
                raise ValueError(
                    "Overlapping buy and sell condition detected.\n"
                    "Please make sure that buy and sell conditions are mutually exclusive."
                )
        self._registered_callbacks['create_signal'] = {
            'buy_condition': buy_condition,
            'sell_condition': sell_condition,
            'first_only': first_only,
        }
        return df
    
    @backtest
    def _open_position(
        self, 
        df: pd.DataFrame,
        order_price: pd.Series | None=None,
        flip_position: bool=True,        
        first_only: bool=True,
    ) -> pd.DataFrame:
        self._registered_callbacks['open_position'] = {
            'order_price': order_price,
            'flip_position': flip_position,
            'first_only': first_only
        }
        return df
    
    @backtest
    def _close_position(
        self, 
        df: pd.DataFrame,
        # TODO: add more time-based options, e.g. 'EOD'?
        when: tSUPPORTED_CLOSE_POSITION_WHEN | list[tSUPPORTED_CLOSE_POSITION_WHEN]=None,
        time_window: int | None=None,
        take_profit: float | None=None,
        stop_loss: float | None=None,
        trailing_stop: float | None=None,
        allow_reopen: bool=True,
    ) -> pd.DataFrame:
        assert any([when, time_window, take_profit, stop_loss, trailing_stop]), \
            "At least one of 'when', 'time_window', 'take_profit', 'stop_loss', 'trailing_stop' must be provided"
        if when:
            if not isinstance(when, list):
                when = [when]
            assert all([w in SUPPORTED_CLOSE_POSITION_WHEN for w in when]), \
                f"Supported 'when' options are {SUPPORTED_CLOSE_POSITION_WHEN}"
        else:
            when = []
        self._registered_callbacks['close_position'] = {
            'when': when,
            'time_window': time_window,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'trailing_stop': trailing_stop,
            'allow_reopen': allow_reopen,
        }
        return df
    
    @backtest
    def _done(self, df: pd.DataFrame) -> pd.DataFrame:
        def _place_orders_and_derive_trades():
            # shift 'order_price' and 'order_size' to the next bar and act as opened limit orders in the same row
            # NOTE: order placed at the end of the previous bar = opened order at the beginning of the next bar
            opened_order_price = grouped_df['order_price'].shift(1)
            opened_order_size = grouped_df['order_size'].shift(1)
            trade_condition = (
                (opened_order_price >= df[['low', 'prev_close', 'next_open']].min(axis=1)) & \
                (opened_order_price <= df[['high', 'prev_close', 'next_open']].max(axis=1)) 
            )
            df['trade_price'] = opened_order_price.where(trade_condition)
            df['trade_size'] = opened_order_size.where(trade_condition)
            if first_trade_only:
                df['_trade_sizes'] = df.groupby(['_trade_streak', '_close_pos'] + self.GROUP)['trade_size'].ffill()
                df['trade_size'] = df['trade_size'].where(grouped_df['_trade_sizes'].diff().ne(0))
                # TODO: clean up orders, e.g. making some orders nan, only need to make order_size nan
            # first_trade_indices = grouped_df.apply(lambda group: group['trade_size'].first_valid_index())
            # last_trade_indices = grouped_df.apply(lambda group: group['trade_size'][::-1].first_valid_index())
            
        self._assert_registered_callbacks()
        
        # Step 1: create signals
        create_signal = self._registered_callbacks['create_signal']
        buy_condition, sell_condition = create_signal['buy_condition'], create_signal['sell_condition']
        if buy_condition is not None and sell_condition is not None:
            conditions = [buy_condition, sell_condition]
            choices = [1, -1]
        else:
            conditions = [buy_condition if buy_condition is not None else sell_condition]
            choices = [1 if buy_condition is not None else -1]
        df['signal'] = np.select(
            conditions,
            choices,
            default=np.nan
        )
        first_signal_only = create_signal['first_only']
        if first_signal_only:
            df['signal'] = df['signal'].where(df.groupby(self.GROUP)['signal'].diff().ne(0))

        if 'open_position' not in self._registered_callbacks:
            return df

        grouped_df = df.groupby(self.GROUP)
        groups = list(df[self.GROUP].drop_duplicates().itertuples(index=False, name=None))
        # add useful columns
        df['_signal_flip'] = grouped_df['signal'].transform(lambda x: x.ffill().diff().ne(0))
        df.loc[df['signal'].isna(), '_signal_flip'] = False
        # set the first True '_signal_flip' to False since theres no signal before it
        first_signal_flip_indices = grouped_df['_signal_flip'].apply(lambda group: group.idxmax())
        for group in groups:
            df.iloc[first_signal_flip_indices.loc[group], df.columns.get_loc('_signal_flip')] = False
        df['_signal_change'] = grouped_df['signal'].transform(lambda x: x.fillna(0).diff().ne(0))
        df['_signal_streak'] = grouped_df['_signal_flip'].cumsum()
        df['_trade_streak'] = grouped_df['_signal_streak'].transform(lambda x: x.shift(1).bfill().astype(int))
       
        
        # Step 2.1: parse 'open_position' and 'close_position'
        open_position = self._registered_callbacks['open_position']
        order_price = open_position['order_price']
        order_price = df['close'] if order_price is None else order_price
        assert ( (order_price > 0) | order_price.isna() ).all(), "'order_price' must be positive or nan"
        first_trade_only = open_position['first_only']
        flip_position = open_position['flip_position']
        # set up orders
        df['order_price'] = np.abs(df['signal']) * order_price
        df['order_size'] = df['signal'] * 1

        df['_close_pos'] = False
        if close_position := self._registered_callbacks.get('close_position', None):
            df['close_remark'] = ''
            
            when: list = close_position['when']
            close_when_signal_flip = 'signal_flip' in when
            close_when_no_signal = 'no_signal' in when
            
            # TODO
            close_at_time_window = close_position['time_window']
            
            # NOTE: only set the close_pos boolean without order_price/size 
            # because the order_size for closing position is unknown
            if close_when_signal_flip:
                mask = grouped_df['_signal_flip'].shift(1)
                df.loc[mask == True, '_close_pos'] = True
                df.loc[mask == True, 'close_remark'] = 'signal_flip'

            if close_when_no_signal:
                mask = grouped_df['_signal_change'].shift(1)
                df.loc[mask == True, '_close_pos'] = True
                df.loc[mask == True, 'close_remark'] = 'no_signal'
            
            # TODO
            allow_reopen = close_position['allow_reopen']
            # df.loc[df['_close_pos'], 'after_close'] = False
            # df['after_close'] = grouped_df['after_close'].ffill()
            # df['after_close'] = (df['after_close'] + ~df['_close_pos']).fillna(False).astype(bool)
        
        
        # Step 2.2: place orders -> trades
        _place_orders_and_derive_trades()
        
        
        # Step 3: add back orders for closing positions, can only be done AFTER having 'trade_size'
        if close_position:
            df['_agg_trade'] = df.groupby(['_trade_streak', '_close_pos'] + self.GROUP)['trade_size'].transform(
                lambda x: x.cumsum().ffill().fillna(0)
            )
            if close_when_signal_flip:
                # close position with close price (market order) when signal flips
                df.loc[df['_signal_flip'], 'order_price'] = df['close']
                # NOTE: trade sizes are now known after step 4, 
                # set back the exact order sizes for closing positions
                df.loc[df['_signal_flip'], 'order_size'] = np.abs(df['_agg_trade']) * df['signal']
            # TODO
            if close_when_no_signal:
                # df.loc[df['_signal_change'], 'order_price'] = df['close']
                pass
            if close_at_time_window:
                pass
            
            # since closing orders have been added, place orders again
            _place_orders_and_derive_trades()
        
        
        # FIXME: should change order size instead? ttl_trade vs agg_trade
        # Step 4: flip position -> change trade_size -> calculate position
        if flip_position:
            df['position_flip'] = grouped_df['trade_size'].transform(lambda x: x.ffill().diff().ne(0))
            df['flip_streak'] = grouped_df['position_flip'].cumsum()
            df['ttl_trade'] = df.groupby(['flip_streak'] + self.GROUP)['trade_size'].cumsum() 
            df['ttl_trade'] = grouped_df['ttl_trade'].transform(
                lambda x: x.ffill().shift(1).where(df['position_flip']).fillna(0)
            )
            df['trade_size'] += np.abs(df['ttl_trade']) * np.sign(df['trade_size'])
        df['position'] = grouped_df['trade_size'].cumsum()
        
        
        # Step 5: add back tp/sl orders for closing position, can only be done AFTER having 'position'
        if close_position:
            # TODO: need to use pfolio to calculate pnls so that we can use take_profit/stop_loss
            # then need to re-calculate position/re-run the whole flow above?
            '''
            current idea:
            -> set close_pos and after_close etc.
            -> go back to step 3 (add stop loss orders)
            -> position + returns
            -> back to step 3, since the returns have changed, may have another stop-loss
            -> position + returns
            ...
            -> loop until it converges, i.e. assert_frame_equal() is True for the last 2 dfs
            '''
            take_profit, stop_loss = close_position['take_profit'], close_position['stop_loss']
            trailing_stop = close_position['trailing_stop']

        
        df['order_price'] = df['order_price'].where(df['order_size'].notna())
        df['trade_price'] = df['trade_price'].where(df['trade_size'].notna())
        
        tmp_cols = [col for col in df.columns if col.startswith('_')]
        df.drop(columns=tmp_cols, inplace=True)
        return df
    
    @backtest
    def preprocess_vectorized_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # add 'prev_close' and 'next_open' columns
        grouped_df = df.groupby(self.GROUP)
        df['prev_close'] = grouped_df['close'].shift(1)
        df['next_open'] = grouped_df['open'].shift(-1)

        # rearrange columns
        left_cols = self.INDEX + ['prev_close']
        core_cols = ['open', 'high', 'low', 'close', 'volume', 'next_open']
        remaining_cols = [col for col in df.columns if col not in left_cols + core_cols]
        df = df.loc[:, left_cols + core_cols + remaining_cols]
        
        # add functions to df
        df.create_signal = lambda *args, **kwargs: self._create_signal(df, *args, **kwargs)
        df.open_position = lambda *args, **kwargs: self._open_position(df, *args, **kwargs)
        df.close_position = lambda *args, **kwargs: self._close_position(df, *args, **kwargs)
        df.done = lambda: self._done(df)
        # TODO: maybe create a subclass like SafeFrame(pd.DataFrame) to prevent users from peeking into the future?
        # e.g. df['close'] = df['close'].shift(-1) should not be allowed
        return df
    
    @staticmethod
    @backtest
    def postprocess_vectorized_df(df_chunks: list[pd.DataFrame]) -> pd.DataFrame:
        '''Processes the df after vectorized backtesting, including:
        - if no 'order_price' column, assumes 'close' as the order price, ≈ market order
        - if no 'order_quantity' column, assumes to be 1
        - 'order_price' will only be looked at after the OHLC bar, 
        i.e. the 'order_price' in the same row 
        not > 'low' and < 'high', the order can't be filled and is considered canceled. ≈ limit order
        - 
        
        - converts orders to trades
        if no 'orders' column, then 'trade_size'/'position' must be in the df
        - derives 'trade_size'/'position' from 'position'/'trade_size'
        '''
        print('***postprocess_vectorized_df!!!')
        # TODO: assert columns exist, e.g. signal, order_price, order_quantity, trade_price, trade_quantity, position
        # TODO: assert order_price and order_quantity not negative
        # TODO: clear columns for debugging, e.g. 'signal1', 'order_price1'
        return
        # df = pd.concat(df_chunks)
        # print(df)
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
    
    @backtest
    def preprocess_event_driven_df(self, df: pd.DataFrame) -> pd.DataFrame:
        def _check_resolution(res):
            from pfund.datas.resolution import Resolution
            resolution = Resolution(res)
            return resolution.is_quote(), resolution.is_tick()
        
        # add 'broker', 'is_quote', 'is_tick' columns
        df['broker'] = df['product'].str.split('-').str[0]
        df['is_quote'], df['is_tick'] = zip(*df['resolution'].apply(_check_resolution))
        
        # arrange columns
        left_cols = self.INDEX + ['broker', 'is_quote', 'is_tick']
        df = df[left_cols + [col for col in df.columns if col not in left_cols]]
        return df
   
    # TODO: 
    @backtest
    def postprocess_event_driven_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert ts column back to datetime type
        pass
    
    # TODO: for train engine
    @train
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
            df.set_index(self.INDEX, inplace=True)
            df.sort_index(level='ts', inplace=True)
            if type_ == 'train':
                self.train_set = df
            elif type_ == 'val':
                self.val_set = self.validation_set = df
            elif type_ == 'test':
                self.test_set = df

       
    '''
    ************************************************
    Helper Functions
    ************************************************
    '''
    @staticmethod
    def output_df_to_parquet(df: pd.DataFrame, file_path: str, compression: str='zstd'):
        df.to_parquet(file_path, compression=compression)

    @staticmethod
    def get_index_values(df: pd.DataFrame, index: str) -> list:
        assert index in df.index.names, f"index must be one of {df.index.names}"
        return df.index.get_level_values(index).unique().to_list()
    
    @staticmethod
    def set_index_values(df: pd.DataFrame, index: str, values: list) -> pd.DataFrame:
        assert index in df.index.names, f"index must be one of {df.index.names}"
        df.index = df.index.set_levels(values, level=index)
        return df
    
    def filter_df_with_multi_index(self, df: pd.DataFrame, start_date: str | None=None, end_date: str | None=None, product: str='', resolution: str=''):
        assert self.INDEX == df.index.names, f"index must be {self.INDEX}"
        product = product or slice(None)
        resolution = resolution or slice(None)
        return df.loc[(slice(start_date, end_date), product, resolution), :]
    
    @staticmethod
    def ffill(df: pd.DataFrame, columns: list[str]):
        return (
            df.unstack(level=columns)
            .ffill()
            .stack(level=columns)
        )
    
    @staticmethod
    def rescale(
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
    