# pyright: reportArgumentType=false, reportUnnecessaryComparison=false, reportOperatorIssue=false, reportAssignmentType=false
from typing import Self

import numpy as np
import pandas as pd


class BacktestDataFrame(pd.DataFrame):
    def create_signal(
        self,
        buy_condition: pd.Series | None=None,
        sell_condition: pd.Series | None=None,
        signal: pd.Series | None=None,
        first_only: bool=False,
    ) -> Self:
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
                raise ValueError("Either buy or sell condition must be provided")
            
            if buy_condition is not None and sell_condition is not None:
                # Validate non-overlapping
                overlaps = buy_condition & sell_condition
                if overlaps.any():
                    raise ValueError(
                        "Overlapping buy and sell condition detected.\n" +
                        "Please make sure that buy and sell conditions are mutually exclusive."
                    )
                conditions = [buy_condition, sell_condition]
                choices = [1, -1]
            else:
                conditions = [buy_condition if buy_condition is not None else sell_condition]
                choices = [1 if buy_condition is not None else -1]

            # Map conditions to signals: buy→1, sell→-1, neither→NaN
            self['signal'] = np.select(conditions, choices, default=np.nan)
        else:
            assert pd.Series(signal.unique()).dropna().isin([1, -1]).all(), "'signal' must only contain 1, -1, nan"
            self['signal'] = signal

        # REVIEW: treat nan as a signal too? useful when e.g. 1 -> 1 -> nan (could be a sell signal) -> 1
        # if is_nan_signal:
        #     df['_signal_change'] = df['signal'].fillna(0).diff().ne(0)
        # else:
        signal_ffill = self["signal"].ffill()
        self["_signal_change"] = signal_ffill.diff().ne(0) & signal_ffill.notna()
        
        # df['_signal_streak'] = df['_signal_change'].cumsum()
        # # signal streak is nan before the first signal occurs
        # df.loc[:first_non_nan_idx-1, '_signal_streak'] = np.nan
        
        if first_only:
            self['signal'] = np.where(self['_signal_change'], self['signal'], np.nan)
        
        return self
    
    def open_position(
        self, 
        order_price: pd.Series | None=None,
        order_quantity: pd.Series | int | float=1,
        first_only: bool=False,
        long_only: bool=False,
        short_only: bool=False,
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
            order_price: price to place the order.
                If None, use 'close' price (market order).
            order_quantity: quantity to place the order.
                If None, use 1.
            first_only: first trade only, do not trade after the first trade until signal changes
            long_only: all order_size in signal=-1 will be set to 0, meaning the order_size will be determined in close_position()
                useful for long-only strategy with signal=-1 to close the position
            short_only: all order_size in signal=1 will be set to 0, meaning the order_size will be determined in close_position()
                useful for short-only strategy with signal=1 to close the position
        '''
        assert 'signal' in self.columns, "No 'signal' column is found, please use create_signal() first"
        assert not (long_only and short_only), "Cannot be long_only and short_only at the same time"
        
        # 1. create orders
        if order_price is None:
            order_price = self['close']
        else:
            assert ( (order_price > 0) | order_price.isna() ).all(), "'order_price' must be positive or nan"
        if isinstance(order_quantity, pd.Series):
            assert ( (order_quantity > 0) | order_quantity.isna() ).all(), "'order_quantity' values must be positive or nan"
        else:
            assert order_quantity > 0, "'order_quantity' must be positive"
        self['order_price'] = np.abs(self['signal']) * order_price
        self['order_size'] = self['signal'] * order_quantity
        
        # 2. place orders
        # shift 'order_price' and 'order_size' to the next bar and act as opened limit orders in the same row
        # NOTE: order placed at the end of the previous bar = opened order at the beginning of the next bar
        opened_order_price = self['order_price'].shift(1)
        opened_order_size = self['order_size'].shift(1)
        opened_order_side = self['signal'].shift(1)

        # 3. fill orders
        # trade_price = min(trade_price, open) if buy, max(trade_price, open) if sell
        prev_close = self['close'].shift(1)
        long_order = (opened_order_side == 1)
        short_order = (opened_order_side == -1)
        # NOTE: here prev_close is used instead of df['open'] to trigger market orders
        # because it's convenient to place market orders by setting order_price=df['close']
        market_order_trade_condition = (
            (long_order & (prev_close <= opened_order_price)) |
            (short_order & (prev_close >= opened_order_price))
        )
        limit_order_trade_condition = (
            (long_order & (opened_order_price >= self['low'])) |
            (short_order & (opened_order_price <= self['high']))
        )
        # NOTE: the actual trade price is 'open', not prev_close
        self['trade_price'] = np.where(
            market_order_trade_condition, 
            self['open'],
            np.where(limit_order_trade_condition, opened_order_price, np.nan)
        )
        trade_condition = market_order_trade_condition | limit_order_trade_condition
        self['trade_size'] = np.where(
            # if order size exceeds volume, trade size = volume * order side
            trade_condition & (opened_order_size.abs() > self['volume']),  
            self['volume'] * opened_order_side, 
            # otherwise trade with order size
            np.where(trade_condition, opened_order_size, np.nan)
        )

        if first_only or long_only or short_only:
            if first_only:
                opposite_side = None
                filtered_orders = pd.Series(True, index=self.index)
            else:  # long_only or short_only with first_only=False:
                opposite_side = -1 if long_only else 1
                filtered_orders = (self['signal'] == opposite_side)
            filtered_trades = filtered_orders.shift(1, fill_value=False if opposite_side is not None else True)

            # To determine the first trade of each trade streak, create trade streaks using trade sides (+1/-1/0))
            # trade side = 0 means no trade but signal changed, used to separate two trade streaks of the same sign but no trades in the middle, 
            # e.g. signal streak: +1, +1 (trade), +1, -1 (no trade, 0 added to trade side), +1, +1 (trade) 
            # without trade_side=0, this example would be treated as one trade streak +1, +1, -1, +1, +1
            trade_price_notna = self['trade_price'].notna()
            self['_trade_side'] = np.where(
                trade_price_notna,
                np.sign(self['trade_size']),
                np.where(self['_signal_change'].shift(1, fill_value=False), 0, np.nan)
            )
            trade_side_ffill = self['_trade_side'].ffill()
            # first trade of the filtered trades
            self['_first_trade'] = (
                trade_side_ffill.diff().ne(0)
                & trade_price_notna  # filter out 0s in self['_trade_side']
                & filtered_trades
            )

            # NOTE: the above is equivalent to the following, which is a groupby version
            # self['_trade_streak'] = self['_signal_streak'].shift(1)
            # self['_first_trade'] = np.where(
            #     self.groupby(['_trade_streak'])['_trade_side'].transform(
            #         lambda x: x.ffill().diff().ne(0)
            #     ) & (self['_trade_side'].notna()) & filtered_trades, 
            #     True, 
            #     False
            # )

            # clean up orders and trades after the first trade
            order_mask = trade_side_ffill.fillna(0).ne(0) & (~self['_signal_change']) & filtered_orders
            self['order_size'] = np.where(order_mask, np.nan, self['order_size'])
            self['order_price'] = np.where(order_mask, np.nan, self['order_price'])
            trade_mask = self['_first_trade'] | ~filtered_trades
            self['trade_size'] = np.where(trade_mask, self['trade_size'], np.nan)
            self['trade_price'] = np.where(trade_mask, self['trade_price'], np.nan)
            
            if long_only or short_only:
                if opposite_side is None:
                    opposite_side = -1 if long_only else 1
                    filtered_orders = (self['signal'] == opposite_side)
                    filtered_trades = filtered_orders.shift(1, fill_value=False)
                # By setting the order size=0, it means the size will be determined in close_position()
                # as the position offset size, i.e. the position will not be flipped when order_size=0
                self['order_size'] = np.where(filtered_orders & (self['order_price'].notna()), 0, self['order_size'])
                self['trade_size'] = np.where(filtered_trades & (self['trade_price'].notna()), 0, self['trade_size'])
           
        return self