from __future__ import annotations
from typing import TYPE_CHECKING, Generator, Literal
if TYPE_CHECKING:
    from pfund.datas.data_base import BaseData

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from pfund.data_tools.data_tool_base import BaseDataTool
from pfund.utils.envs import backtest, train


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
        signal: pd.Series | None=None,
        first_only: bool=False,
    ) -> pd.DataFrame:
        '''
        A signal is defined as a sequence of 1s and -1s, where 1 means a buy signal and -1 means a sell signal.
        Args:
            buy_condition: condition to create a buy signal 1
            sell_condition: condition to create a sell signal -1
            signal: provides self-defined signals, buy_condition and sell_condition are ignored if provided
            first_only: only the first signal is remained in each signal sequence
                useful when only the first signal is treated as a true signal
        '''
        if signal is None:
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
        else:
            assert np.isin(signal.unique(), [1, -1, np.nan]).all(), "'signal' must only contain 1, -1, nan"
            df['signal'] = signal
            
        # if is_nan_signal:
        #     df['_signal_change'] = df['signal'].fillna(0).diff().ne(0)
        # else:
        df['_signal_change'] = df['signal'].ffill().diff().ne(0)
        
        first_non_nan_idx = df['signal'].first_valid_index()
        
        # set the first nan sequence to False
        df.loc[:first_non_nan_idx-1, '_signal_change'] = False
        
        # df['_signal_streak'] = df['_signal_change'].cumsum()
        # # signal streak is nan before the first signal occurs
        # df.loc[:first_non_nan_idx-1, '_signal_streak'] = np.nan
        
        if first_only:
            df['signal'] = np.where(df['_signal_change'], df['signal'], np.nan)
        
        return df
    
    @backtest
    def _open_position(
        self, 
        df: pd.DataFrame,
        order_price: pd.Series | None=None,
        order_quantity: pd.Series | None=None,
        first_only: bool=False,
        long_only: bool=False,
        short_only: bool=False,
        ignore_sizing: bool=True,
        fill_ratio: float=0.1,
    ) -> pd.DataFrame:
        '''
        Opens positions in a vectorized manner.
        Conceptually, this function places orders at the end of bar/candlestick N.
        For example, for a buy order:
        - If the order price >= close price of bar N, it is a market order,
            by assuming that the close price is the current best price; otherwise it is a limit order.
        Then the orders are opened at the beginning of bar N+1,
        and filled in the duration of bar N+1, if high >= order price >= low.
        Opened orders are considered as cancelled at the end of bar N+1 if not filled.
        Args:
            ignore_sizing: if True, ignore the 'order_quantity' and 'fill_ratio' and just use order quantity=1
            first_only: first trade only, do not trade after the first trade until signal changes
            long_only: all order_size in signal=-1 will be set to 0, meaning the order_size will be determined in close_position()
                useful for long-only strategy with signal=-1 to close the position
            short_only: all order_size in signal=1 will be set to 0, meaning the order_size will be determined in close_position()
                useful for short-only strategy with signal=1 to close the position
            fill_ratio: fill_ratio * volume = max liquidity for each trade
        '''
        assert 'signal' in df.columns, "No 'signal' column is found, please use create_signal() first"
        assert not (long_only and short_only), "Cannot be long_only and short_only at the same time"
        
        assert 1 >= fill_ratio > 0, "'fill_ratio' must be between 0 and 1"
        
        # 1. create orders
        if order_price is None:
            order_price = df['close']
        else:
            assert ( (order_price > 0) | order_price.isna() ).all(), "'order_price' must be positive or nan"
        if order_quantity is None or ignore_sizing:
            order_quantity = 1
        else:
            assert ( (order_quantity > 0) | order_quantity.isna() ).all(), "'order_quantity' must be positive or nan"
        df['order_price'] = np.abs(df['signal']) * order_price
        df['order_size'] = df['signal'] * order_quantity
        
        # 2. place orders
        # shift 'order_price' and 'order_size' to the next bar and act as opened limit orders in the same row
        # NOTE: order placed at the end of the previous bar = opened order at the beginning of the next bar
        opened_order_price = df['order_price'].shift(1)
        opened_order_size = df['order_size'].shift(1)
        opened_order_side = df['signal'].shift(1)

        # 3. fill orders
        # trade_price = min(trade_price, open) if buy, max(trade_price, open) if sell
        prev_close = df['close'].shift(1)
        long_order = (opened_order_side == 1)
        short_order = (opened_order_side == -1)
        # NOTE: here prev_close is used instead of df['open'] to trigger market orders
        # because it's convenient to place market orders by setting order_price=df['close']
        market_order_trade_condition = (
            (long_order & (prev_close <= opened_order_price)) |
            (short_order & (prev_close >= opened_order_price))
        )
        limit_order_trade_condition = (
            (long_order & (opened_order_price >= df['low'])) |
            (short_order & (opened_order_price <= df['high']))
        )
        # NOTE: the actual trade price is 'open', not prev_close
        df['trade_price'] = np.where(
            market_order_trade_condition, 
            df['open'],  # df['open'] * (1 + slippage * opened_order_side),
            np.where(limit_order_trade_condition, opened_order_price, np.nan)
        )
        
        trade_condition = market_order_trade_condition | limit_order_trade_condition
        if ignore_sizing:
            df['trade_size'] = np.where(trade_condition, opened_order_size, np.nan)
        else:
            # make sure the trade quantity does not exceed 'volume' * fill_ratio
            max_liquidity = df['volume'] * fill_ratio
            use_capped_liquidity = trade_condition & (opened_order_size.abs() > max_liquidity)
            df['trade_size'] = np.where(
                use_capped_liquidity, max_liquidity * opened_order_side, 
                np.where(trade_condition, opened_order_size, np.nan)
            )
            
        if first_only or long_only or short_only:
            if first_only:
                opposite_side = None
                filtered_orders = pd.Series(True, index=df.index)
            else:  # long_only or short_only:
                # only applies first_trade to the opposite side
                # e.g. if long_only, only the first trade of -1 is remained and the rest are set to nan
                opposite_side = -1 if long_only else 1
                filtered_orders = (df['signal'] == opposite_side)
            filtered_trades = filtered_orders.shift(1, fill_value=False if opposite_side is not None else True)

            # NOTE: whenever there is a signal change and no trade after it, add a 0 to _trade_side.
            # used to separate trade streaks with the same sign but no trades in the middle
            # e.g. +1, +1 (trade), +1, -1 (no trade, 0 added to trade side), +1, +1 (trade), ..., 
            # so that the two +1 trade streaks are separated and ffill().diff().ne(0) works correctly
            trade_price_notna = df['trade_price'].notna()
            df['_trade_side_with_0s'] = np.where(
                trade_price_notna,
                np.sign(df['trade_size']),
                np.where(df['_signal_change'].shift(1, fill_value=False), 0, np.nan)
            )
            trade_side_with_0s_ffill = df['_trade_side_with_0s'].ffill()
            df['_first_trade'] = (
                trade_side_with_0s_ffill.diff().ne(0)
                & trade_price_notna  # filter out 0s in _trade_side_with_0s
                & filtered_trades
            )
            
            # NOTE: the above is equivalent to the following, which is a groupby version
            # df['_trade_streak'] = df['_signal_streak'].shift(1)
            # df['_first_trade'] = np.where(
            #     df.groupby(['_trade_streak'])['_trade_side'].transform(
            #         lambda x: x.ffill().diff().ne(0)
            #     ) & (df['_trade_side'].notna()) & filtered_trades, 
            #     True, 
            #     False
            # )

            # clean up orders and trades after the first trade
            first_trade_forwards_mask = trade_side_with_0s_ffill.fillna(0).ne(0) 
            order_mask = first_trade_forwards_mask & (~df['_signal_change']) & filtered_orders
            df['order_size'] = np.where(order_mask, np.nan, df['order_size'])
            df['order_price'] = np.where(order_mask, np.nan, df['order_price'])
            trade_mask = df['_first_trade'] | ~filtered_trades
            df['trade_size'] = np.where(trade_mask, df['trade_size'], np.nan)
            df['trade_price'] = np.where(trade_mask, df['trade_price'], np.nan)
            
            if long_only or short_only:
                if opposite_side is None:
                    opposite_side = -1 if long_only else 1
                    filtered_orders = (df['signal'] == opposite_side)
                    filtered_trades = filtered_orders.shift(1, fill_value=False)
                # By setting the order size=0, it means the size will be determined in close_position()
                # as the position offset size, i.e. the position will not be flipped when order_size=0
                df['order_size'] = np.where(filtered_orders & (df['order_price'].notna()), 0, df['order_size'])
                df['trade_size'] = np.where(filtered_trades & (df['trade_price'].notna()), 0, df['trade_size'])
        
        return df
    
    @backtest
    def _close_position(
        self, 
        df: pd.DataFrame,
        take_profit: float | None=None,
        stop_loss: float | None=None,
    ) -> pd.DataFrame:
        '''
        Closes positions in a vectorized manner.
        Conceptually, this function places stop market orders at the end of bar N, after placing orders in _open_position().
        Due to limitation #6, only one trade can be created at a time, by either an opened order or a stop order.
        Therefore, trade_price and trade_size can be updated by stop orders, not just opened orders.
        
        Since whether 'high' or 'low' is reached first cannot be known,
        it is always assumed that 'low' is reached first in long position and 'high' is reached first in short position
        so that stop loss is prioritized over take profit.
        
        Limitations:
        1. position side must change after signal has changed, or in other words,
            the position must be closed or flipped after signal has changed.
            Reason:
                When this statement holds, the following formula is true for calculating the average price for each trade streak:
                avg_price = (df['trade_price'] * df['trade_size']).cumsum() / df['trade_size'].cumsum()
            Explanation:
                For example, if there are trades: +1, +1, +1, -1 (signal change), ...
                where (+1, +1, +1) is trade streak #1 and the position at signal change is +2.
                Then the avg_price formula mentioned above does not hold,
                because avg_price is still the avg_price of trade streak #1 without the -1 trade,
                and the -1 trade does not contribute to the avg_price calculation but only realizes the profit/loss of the position.
                Therefore, we need to make sure that the position has flipped or closed before calculating the avg_price.
                From then on, we can also calculate stop-loss/take-profit correctly.
        2. To ensure #1 is always true and the position can always be closed, 
            the actual total traded volume available is ignored when closing position.
        3. position cannot be re-entered after being closed by stop-loss/take-profit in the same trade streak
            This is a huge limitation for long-only/short-only strategies with only +1s/only -1s signals.
            For example, if the strategy is long-only and depends on stop-loss to close the position,
            a for loop is needed since #1 no longer holds in this case.
            However, if this long-only strategy has prepared -1 signals in advance,
            instead of relying on stop-loss to close the position,
            then #1 still applies.
        4. stop_loss/take_profit only supports stop market orders, because the exact price movement after stop_loss/take_profit is unknown,
            placing limit orders might not get filled.
        5. assumes stop loss is always triggered before take profit if the order of happening cannot be determined.
            - in long position, check 'low' first; in short position, check 'high' first
        6. trade priority of opened order, stop loss order and take profit order, only one of them can be traded at a time.
        - e.g. if long opened order and long position,
            1. if stop order is triggered immediately when placed, then it has the priority over market order
            2. if opened order is a market order, then it will be traded.
            3. if opened order is a limit order, then it will be traded first if order_price > stop_loss_price.
            4. if stop loss order is not triggered, then check take profit order
        '''
        def _calculate_avg_price():
            df['_agg_costs'] = (
                df
                .assign(_cost=lambda x: x['trade_price'] * x['trade_size'].fillna(0))
                .groupby(['_position_streak'])['_cost'].cumsum()
            )
            df['avg_price'] = np.where(df['position'] != 0, df['_agg_costs'] / df['position'], np.nan)
            df['avg_price'] = df['avg_price'].ffill()
            
        def _calculate_stop_price():
            '''Calculates stop price for stop-loss and take-profit at the same time,
            only triggered stop prices are shown.
            Keeps the first stop order for each position streak,
            updates the trade_size and trade_price accordingly,
            and clean up orders and trades after stop orders.
            '''
            end_position_side = np.sign(df['position'])
            start_position_side = end_position_side.shift(1, fill_value=0)
            prev_close = df['close'].shift(1)
            opened_order_side = df['signal'].shift(1)
            opened_order_price = df['order_price'].shift(1)
            long_order = (opened_order_side == 1)
            short_order = (opened_order_side == -1)
            market_order_trade_condition = (
                (long_order & (prev_close <= opened_order_price)) |
                (short_order & (prev_close >= opened_order_price))
            )
            limit_order_trade_condition = (
                (long_order & (opened_order_price >= df['low'])) |
                (short_order & (opened_order_price <= df['high']))
            )
            
            for tp_or_sl, sign in [(stop_loss, -1), (take_profit, 1)]:
                if tp_or_sl is None:
                    continue
                stop_price = df['avg_price'] * (1 + end_position_side * tp_or_sl * sign)
                opened_stop_order_price = stop_price.shift(1)
                positive_sign = (start_position_side * sign == 1)
                negative_sign = (start_position_side * sign == -1)

                # stop orders that are triggered immediately
                stop_market_order_trade_condition_immediate = (
                    (negative_sign & (prev_close <= opened_stop_order_price)) |
                    (positive_sign & (prev_close >= opened_stop_order_price))    
                )
                
                stop_market_order_trade_condition = (
                    (negative_sign & (df['low'] <= opened_stop_order_price)) |
                    (positive_sign & (df['high'] >= opened_stop_order_price))
                )
                
                is_limit_order_filled_first = (
                    limit_order_trade_condition
                    & (
                       negative_sign & long_order & (opened_order_price >= opened_stop_order_price) |
                       positive_sign & short_order & (opened_order_price <= opened_stop_order_price)
                    )
                )
                
                stop_market_order_trade_condition = np.where(
                    stop_market_order_trade_condition_immediate,
                    True,
                    np.where(
                        stop_market_order_trade_condition & (~market_order_trade_condition) & (~is_limit_order_filled_first),
                        True,
                        False
                    )
                )
                
                # only kept those triggered stop prices
                df['stop_price'] = np.where(
                    # needs df['stop_price'].isna() so that the first stop price is kept, i.e. stop loss won't be overridden by take profit
                    stop_market_order_trade_condition & df['stop_price'].isna(),  
                    opened_stop_order_price,
                    df['stop_price']
                )
            df['stop_price'] = df['stop_price'].shift(-1)  # shift back stop_price from trade row to order row
      
            # NOTE: applying the same logic as _first_trade in open_position()
            stop_price_notna = df['stop_price'].notna()
            df['_stop_side_with_0s'] = np.where(
                stop_price_notna,
                end_position_side,
                np.where(df['_position_change'], 0, np.nan)
            )
            stop_side_with_0s_ffill = df['_stop_side_with_0s'].ffill()
            df['_first_stop_order'] = (
                stop_side_with_0s_ffill.diff().ne(0) 
                & stop_price_notna  # filter out 0s in _stop_side_with_0s
            )

            # clean up stop_price, only the first one in each streak is left
            df['stop_price'] = np.where(df['_first_stop_order'], df['stop_price'], np.nan)
            price_diff_check = (end_position_side * (df['avg_price'] - df['stop_price'])).shift(1)
            df['_stop_loss'] = np.where(price_diff_check > 0, True, False)
            df['_take_profit'] = np.where(price_diff_check < 0, True, False)
            
            # update trades created by stop orders
            offset_size = df['position'].shift(1, fill_value=0) * (-1)
            first_stop_trade = df['_first_stop_order'].shift(1, fill_value=False)
            df['trade_size'] = np.where(first_stop_trade, offset_size, df['trade_size'])
            df['trade_price'] = np.where(
                first_stop_trade, 
                df['stop_price'].shift(1),  # df['stop_price'].shift(1) * (1 - start_position_side * slippage),
                df['trade_price']
            )

            # clean up order, trades and 'position' after stop orders
            first_stop_order_forwards_mask = stop_side_with_0s_ffill.fillna(0).ne(0) 
            order_mask = first_stop_order_forwards_mask & (df['signal'] == np.sign(stop_side_with_0s_ffill)) & df['stop_price'].isna()
            df['order_size'] = np.where(order_mask, np.nan, df['order_size'])
            df['order_price'] = np.where(order_mask, np.nan, df['order_price'])
            position_mask = first_stop_order_forwards_mask & (~df['_first_stop_order'])
            df['position'] = np.where(position_mask, 0.0, df['position'])
            trade_mask = position_mask & (~first_stop_trade)
            df['trade_size'] = np.where(trade_mask, np.nan, df['trade_size'])
            df['trade_price'] = np.where(trade_mask, np.nan, df['trade_price'])
        
        if take_profit:
            assert take_profit > 0, "'take_profit' must be positive"
        if stop_loss:
            stop_loss = abs(stop_loss)
            assert 1 > stop_loss > 0, "'stop_loss' must be between 0 and 1"
        
        trade_side = np.sign(df['trade_size'])

        # position change = position closed or flipped
        df['_position_change'] = trade_side.ffill().diff().ne(0) & trade_side.notna()
        # NOTE: position streak also includes opposite signals if not traded,
        # e.g. +1, +1, +1, -1 (no trade), -1 (no trade), ...
        df['_position_streak'] = df['_position_change'].cumsum()
        df['position'] = df.groupby('_position_streak')['trade_size'].transform(
            lambda x: x.cumsum().ffill()
        ).fillna(0)
        
        _calculate_avg_price()
        df['stop_price'] = np.nan
        if take_profit or stop_loss:
            _calculate_stop_price()
        df['avg_price'] = np.where(df['position'] == 0, np.nan, df['avg_price'])
        
        offset_order_size = df['position'] * (-1)
        offset_trade_size = offset_order_size.shift(1, fill_value=0)
        # update trade_size and order_size
        df['trade_size'] = np.where(
            df['_position_change'] & (offset_trade_size != 0) & df['stop_price'].shift(1).isna(), 
            offset_trade_size + df['trade_size'], 
            df['trade_size']
        )
        df['order_size'] = np.where(
            (np.sign(offset_order_size) == df['signal']) & (offset_order_size != 0),
            offset_order_size + df['order_size'], 
            df['order_size']
        )
        if df['stop_price'].isna().all():
            df.drop(columns=['stop_price'], inplace=True)
            
        return df
    
    @backtest
    def preprocess_vectorized_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # rearrange columns
        left_cols = self.INDEX
        core_cols = ['open', 'high', 'low', 'close', 'volume']
        remaining_cols = [col for col in df.columns if col not in left_cols + core_cols]
        df = df.loc[:, left_cols + core_cols + remaining_cols]
        
        # add functions to df
        df.create_signal = lambda *args, **kwargs: self._create_signal(df, *args, **kwargs)
        df.open_position = lambda *args, **kwargs: self._open_position(df, *args, **kwargs)
        df.close_position = lambda *args, **kwargs: self._close_position(df, *args, **kwargs)

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
        # TODO: assert columns exist, e.g. signal, order_price, order_quantity, trade_price, trade_quantity, position
        
        def _clean_up_columns(df: pd.DataFrame) -> pd.DataFrame:
            '''Clean up temporary columns that start with '_' and add 'remark' column for debugging.
            Args:
                remark_cols: columns to add to 'remark'
                debug_cols: temporary columns to keep for debugging
            '''
            remark_cols = ['_signal_change', '_first_trade', '_position_change', '_stop_loss', '_take_profit']
            aliases = {
                'signal_change': 'sc',
                'first_trade': 'ft',
                'position_change': 'pc',
                'stop_loss': 'sl',
                'take_profit': 'tp',
            }
            df['remark'] = ''
            for col in remark_cols:
                if col in df.columns:
                    df.loc[df[col], 'remark'] += f',{aliases[col[1:]]}'  # [1:] removes leading '_'
            df['remark'] = df['remark'].str.lstrip(',')  # remove leading comma 
            
            # remove temporary columns that start with '_'
            # TEMP
            debug_cols = ['_position_change', '_stop_loss', '_take_profit']
            remove_cols = [col for col in df.columns if col.startswith('_') and col not in debug_cols]
            df.drop(columns=remove_cols, inplace=True)
            return df
        
        # TODO: add order_type: 'market'/'limit'/'stop'
        df = pd.concat(df_chunks)
        df = _clean_up_columns(df)
        return df
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
    def output_df_to_parquet(df: pd.DataFrame, file_path: str | Path, compression: str='zstd'):
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
    