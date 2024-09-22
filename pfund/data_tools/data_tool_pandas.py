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


SUPPORTED_CLOSE_POSITION_WHEN = ['signal_change']
tSUPPORTED_CLOSE_POSITION_WHEN = Literal['signal_change']


cprint = lambda msg: Console().print(msg, style='bold')


# NOTE: convention: all function names that endswith "_df" will directly modify self.df, e.g. "xxx_df"
class PandasDataTool(BaseDataTool):
    def __init__(self):
        super().__init__('pandas')
    
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
        if first_only:
            df['signal'] = df['signal'].where(df['signal'].diff().ne(0))
        return df
    
    @backtest
    def _open_position(
        self, 
        df: pd.DataFrame,
        order_price: pd.Series | None=None,
        flip_position: bool=True,        
        first_only: bool=True,
        is_nan_signal: bool=False,
    ) -> pd.DataFrame:
        '''
        Args:
            is_nan_signal: Only matters when first_only is True,
            - if True, nans are also considered as signals, e.g. 1(trade),1,1,nan,nan,1 (trade),1,1
                The 2nd 1s sequence can trade again when first_only=True 
                because the nans between are considered as a new signal trend
            - if False, nans are ignored when considering first only trade, e.g. 1(trade),1,1,nan,nan,1,1,1
        '''
        self._registered_callbacks['open_position'] = {
            'order_price': order_price,
            'flip_position': flip_position,
            'first_only': first_only,
            'is_nan_signal': is_nan_signal,
        }
        return df
    
    @backtest
    def _close_position(
        self, 
        df: pd.DataFrame,
        when: tSUPPORTED_CLOSE_POSITION_WHEN | None='signal_change',
        take_profit: float | None=None,
        stop_loss: float | None=None,
        time_window: int | None=None,
        trailing_stop: float | None=None,
    ) -> pd.DataFrame:
        assert any([when, take_profit, stop_loss, time_window, trailing_stop]), \
            "At least one of 'when', 'take_profit', 'stop_loss', 'time_window', 'trailing_stop' must be provided"
        if when:
            assert when in SUPPORTED_CLOSE_POSITION_WHEN, \
                f"Supported 'when' options are {SUPPORTED_CLOSE_POSITION_WHEN}"
        if take_profit:
            assert take_profit > 0, "'take_profit' must be positive"
        if stop_loss:
            stop_loss = abs(stop_loss)
            assert 1 > stop_loss > 0, "'stop_loss' must be between 0 and 1"
        # TODO:
        if time_window:
            assert time_window > 0 and isinstance(time_window, int), "'time_window' must be a positive integer"
            raise NotImplementedError("time_window > 1 is not yet implemented")
        # TODO:
        if trailing_stop:
            raise NotImplementedError("trailing_stop is not yet implemented")
        self._registered_callbacks['close_position'] = {
            'when': when,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'time_window': time_window,
            'trailing_stop': trailing_stop,
        }
        return df
    
    @backtest
    def _done(self, df: pd.DataFrame, debug=False) -> pd.DataFrame:
        '''Orchestrate the whole process of opening and closing positions after signals are created.
        '''
        def _set_first_value(col: str, value):
            first_index = df[col].idxmax()
            df.iloc[first_index, df.columns.get_loc(col)] = value
            # GROUPING version:
            # grouped_df = df.groupby(self.GROUP)
            # groups = list(df[self.GROUP].drop_duplicates().itertuples(index=False, name=None))
            # first_indices = grouped_df[col].apply(lambda group: group.idxmax())
            # for group in groups:
            #     df.iloc[first_indices.loc[group], df.columns.get_loc(col)] = value

        def _open_position():
            '''
            Dependency columns: ['order_price', 'order_size']
            Updated columns: ['trade_price', 'trade_size']
            '''
            # shift 'order_price' and 'order_size' to the next bar and act as opened limit orders in the same row
            # NOTE: order placed at the end of the previous bar = opened order at the beginning of the next bar
            opened_order_price = df['order_price'].shift(1)
            opened_order_size = df['order_size'].shift(1)
            trade_condition = (
                (opened_order_price >= df[['low', 'prev_close', 'next_open']].min(axis=1)) & \
                (opened_order_price <= df[['high', 'prev_close', 'next_open']].max(axis=1)) 
            )
            df['trade_price'] = opened_order_price.where(trade_condition)
            df['trade_size'] = opened_order_size.where(trade_condition)
            
            if first_trade_only:
                df['agg_trade'] = df.groupby(['_trade_streak'])['trade_size'].ffill()
                df['trade_size'] = df['trade_size'].where(df['agg_trade'].diff().ne(0))
                
                # clean up orders (order_size) after the first trade
                after_trade_mask = df.groupby(['_signal_streak'])['trade_size'].transform(
                    lambda x: x.fillna(0).cumsum().astype(bool)
                )
                df.loc[after_trade_mask, 'order_size'] = np.nan
        
        def _close_position():
            '''
            Add back orders for closing position.
            Dependency columns: ['trade_size']
            Updated columns: ['position_close', 'trade_price', 'trade_size', 'order_price', 'order_size']
            '''
            mask = (df[f'_{close_when}'].shift(1) == True)
            df.loc[mask, 'position_close'] = True

            df['_close_streak'] = df['position_close'].cumsum()
            # if there happens to be a trade overlapping with the closing position, make it nan
            # so that 'agg_trade' is not affected by the trade
            df.loc[df['position_close'], 'trade_price'] = np.nan
            df.loc[df['position_close'], 'trade_size'] = np.nan
            df['agg_trade'] = df.groupby(['_close_streak'])['trade_size'].transform(
                lambda x: x.cumsum().ffill().fillna(0)
            )

            # set 'position_close'=False if no trade
            df.loc[df['agg_trade'].shift(1) == 0, 'position_close'] = False

            df.loc[df['position_close'], 'trade_price'] = df['close'].shift(1)
            df.loc[df['position_close'], 'trade_size'] =  df['agg_trade'].shift(1) * (-1)
            
            # update the orders
            df.loc[df['position_close'].shift(-1, fill_value=False), 'order_price'] = df['trade_price'].shift(-1)
            df.loc[df['position_close'].shift(-1, fill_value=False), 'order_size'] = df['trade_size'].shift(-1)
            
        def _flip_position():
            '''
            Flip position by changing 'trade_size'
            Dependency columns: ['trade_size']
            Updated columns: ['trade_size', 'order_size']
            '''
            df['position_flip'] = df['trade_size'].ffill().diff().fillna(False).ne(0)
            df.loc[df['position_close'], 'position_flip'] = False
            df['flip_streak'] = df['position_flip'].cumsum()
            
            df['agg_trade'] = df.groupby(['flip_streak'])['trade_size'].transform(
                lambda x: x.cumsum().ffill().fillna(0)
            )
            # set 'position_flip'=False if no trade
            df.loc[df['agg_trade'].shift(1) == 0, 'position_flip'] = False

            # update trade_size for position flip, not orders because unlike closing position, 
            # we don't need to update the order_price as well, so we can just update trade_size -> order_size
            df.loc[df['position_flip'], 'trade_size'] += df['agg_trade'].shift(1) * (-1)
            
            # update back order_size for reference after changing trade_size
            after_trade_mask = df.groupby(['_signal_streak'])['trade_size'].transform(
                lambda x: x.fillna(0).cumsum().astype(bool)
            )
            df.loc[~after_trade_mask & df['order_price'].notna(), 'order_size'] = df['trade_size'].bfill()
        
        def _calculate_position():
            df['position'] = df['trade_size'].transform(
                lambda x: x.cumsum().ffill().fillna(0).astype(int)
            )
        
        def _calculate_avg_price():
            df['_avg_streak'] = df['_trade_streak'].shift(1).bfill().astype(int)
            df['_cost'] = (df['trade_price'] * df['trade_size']).fillna(0)
            df['agg_costs'] = df.groupby(['_avg_streak'])['_cost'].cumsum()
            df['agg_trade'] = df.groupby(['_avg_streak'])['trade_size'].transform(
                lambda x: x.cumsum().ffill().fillna(0)
            )
            df['avg_price'] = np.where(df['agg_trade'] != 0, df['agg_costs'] / df['agg_trade'], np.nan)
            df['avg_price'] = df.groupby('_avg_streak')['avg_price'].ffill()    
        
        assert 'signal' in df.columns, "No 'signal' column is found, please use create_signal() first"
        if 'close_position' in self._registered_callbacks:
            assert 'open_position' in self._registered_callbacks, "No 'open_position' callback is registered"
        elif 'open_position' not in self._registered_callbacks:
            return df

        use_for_loop = False
        
        # parse registered callbacks
        open_position = self._registered_callbacks['open_position']
        order_price = open_position['order_price']
        order_price = df['close'] if order_price is None else order_price
        assert ( (order_price > 0) | order_price.isna() ).all(), "'order_price' must be positive or nan"
        is_nan_signal = open_position['is_nan_signal']
        first_trade_only = open_position['first_only']
        flip_position = open_position['flip_position']
        if close_position := self._registered_callbacks.get('close_position', None):
            close_when: str = close_position['when']
            take_profit, stop_loss = close_position['take_profit'], close_position['stop_loss']
            time_window = close_position['time_window']
            trailing_stop = close_position['trailing_stop']
            if not (close_when and (take_profit or stop_loss or time_window or trailing_stop)):
                cprint("Position dependencies detected, using FOR LOOP in vectorized backtesting!")
                use_for_loop = True
        
        # set up orders
        df['order_price'] = np.abs(df['signal']) * order_price
        df['order_size'] = df['signal'] * 1
        
        # create common columns and variables
        if not is_nan_signal:
            df['_signal_change'] = df['signal'].ffill().diff().ne(0)
            df.loc[df['signal'].isna(), '_signal_change'] = False
        else:
            # treat nans as signals
            df['_signal_change'] = df['signal'].fillna(0).diff().fillna(0).ne(0)
        # set the first True value to False since theres no signal before it
        _set_first_value(col='_signal_change', value=False)
        df['_signal_streak'] = df['_signal_change'].cumsum()
        df['_trade_streak'] = df['_signal_streak'].shift(1).bfill().astype(int)
        df['position_close'] = False
        df['position_flip'] = False
        
        
        if not use_for_loop:
            _open_position()
            if close_position:
                if close_when:
                    _close_position()
                    _calculate_avg_price()
                    if stop_loss:
                        df['sl_price'] = np.where(
                            ~df['position_close'],
                            df['avg_price'] * (1 - np.sign(df['trade_size'].ffill()) * stop_loss),
                            np.nan
                        )
                        df['sl_price'] = df['sl_price'].shift(1)
                        trigger_condition = (df['high'] >= df['sl_price']) & (df['sl_price'] >= df['low'])
                        df['sl_price'] = df['sl_price'].where(trigger_condition & ~df['position_close'])
                        df['_sl_close'] = df.groupby('_avg_streak')['sl_price'].transform(
                            lambda x: x.ffill().fillna(0).astype(bool).diff().fillna(0).ne(0)
                        )
                        df.loc[df['_sl_close'], 'trade_size'] = df['agg_trade'].shift(1) * (-1)
                        # update back the trade_size and order_size after stop loss
                        after_sl_mask = df.groupby(['_avg_streak'])['_sl_close'].transform(
                            lambda x: x.fillna(0).cumsum().astype(bool)
                        )
                        df.loc[after_sl_mask & ~df['_sl_close'], 'trade_size'] = np.nan
                        df.loc[after_sl_mask & ~df['position_close'], 'order_size'] = np.nan
                        
                    if take_profit:
                        df['tp_price'] = np.where(
                            ~df['position_close'],
                            df['avg_price'] * (1 + np.sign(df['trade_size'].ffill()) * take_profit),
                            np.nan
                        )
                        trigger_condition = (df['high'] >= df['tp_price']) & (df['tp_price'] >= df['low'])
                        df['tp_price'] = df['tp_price'].where(trigger_condition)
                        
                    # TODO: 'first' will make the order of take_profit and  stop_loss different
                    import random
                    random.seed(8)
                    # df['first'] = random.choices(['H', 'L', 'N'], k=df.shape[0])
                   
            if flip_position:
                _flip_position()
            _calculate_position()
        else:
            # TODO: consider all: first_trade_only, flip_position, close_position ...
            raise NotImplementedError("The for-loop approach is not yet implemented for vectorized backtesting")
        
        
        # clean up 'order_price'/'trade_price' if the corresponding 'order_size'/trade_size' is removed
        df['order_price'] = df['order_price'].where(df['order_size'].notna())
        df['trade_price'] = df['trade_price'].where(df['trade_size'].notna())


        # add remarks
        if debug:
            df['remark'] = ''
            df.loc[df['_signal_change'], 'remark'] += ',signal_change'
            if first_trade_only:
                df.loc[df['trade_size'].notna() & (~df['position_close']), 'remark'] += ',first_trade'
            df.loc[df['position_close'], 'remark'] += ',position_close'
            df.loc[df['_sl_close'], 'remark'] += ',sl_close'
            df.loc[df['position_flip'], 'remark'] += ',position_flip'
            # remove leading comma
            df['remark'] = df['remark'].str.lstrip(',')
            

        # check if positions are closed and flipped correctly
        if flip_position:
            if position_after_flipped := df.loc[df['position_flip'], 'position'].unique():
                # if only 1s and -1s in position_after_flipped, this will make only 1 left
                position_after_flipped = np.unique(np.abs(position_after_flipped))
                assert position_after_flipped.shape[0] == 1 and position_after_flipped[0] == 1, \
                    f"position_after_flipped={position_after_flipped}"
        if close_position:
            if position_after_closed := df.loc[df['position_close'], 'position'].unique():
                assert position_after_closed.shape[0] == 1 and position_after_closed[0] == 0, \
                    f"position_after_closed={position_after_closed}"
        

        # remove temporary columns
        # TODO: when stable, remove 'order_price' and 'order_size' as well, i.e. rename to '_order_price' and '_order_size'
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
        df.done = lambda *args, **kwargs: self._done(df, *args, **kwargs)

        # set alias
        if not hasattr(df, 'create'):
            df.create = df.create_signal
        if not hasattr(df, 'open'):
            df.open = df.open_position
        if not hasattr(df, 'close'):
            df.close = df.close_position
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
    