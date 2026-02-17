# pyright: reportArgumentType=false, reportUnnecessaryComparison=false, reportOperatorIssue=false, reportAssignmentType=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportGeneralTypeIssues=false
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
    ) -> Self:
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
        # NOTE: ignore the actual traded 'volume', just use the order size directly
        self['trade_size'] = np.where(trade_condition, opened_order_size, np.nan)
        
        # this version takes the actual traded 'volume' into account
        # self['trade_size'] = np.where(
        #     # if order size exceeds volume, trade size = volume * order side
        #     trade_condition & (opened_order_size.abs() > self['volume']),  
        #     self['volume'] * opened_order_side, 
        #     # otherwise trade with order size
        #     np.where(trade_condition, opened_order_size, np.nan)
        # )

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
            # NOTE: this trade_side has a bunch of 0s added to separate two trade streaks (so that order_mask below can work) of the same sign but no trades in the middle
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
    
    def close_position(
        self,
        take_profit: float | None=None,
        stop_loss: float | None=None,
        time_window: int | None=None,
    ) -> Self:
        '''
        Closes positions in a vectorized manner.
        Conceptually, this function places stop market orders at the end of each bar, after placing orders in open_position().
        Due to limitation #6, only one trade can be created at a time, by either an opened order or a stop order.
        Therefore, trade_price and trade_size can be updated by stop orders, not just opened orders.
        
        Limitations:
        1. position side must change after signal has changed, or in other words,
            the position must be closed or flipped after signal has changed.
            Reason:
                When this statement holds, the following formula is true for calculating the average price for each trade streak:
                avg_price = (df['trade_price'] * df['trade_size']).cumsum() / df['trade_size'].cumsum()
            Explanation:
                For example, if there are trades: +1, +1, +1, -1 (signal change), ...
                where (+1, +1, +1) is trade streak #1 and the position at signal change is +2 (+1+1+1-1 = +2).
                Then the avg_price formula mentioned above does not hold,
                because avg_price is still the avg_price of trade streak #1 without the -1 trade,
                and the -1 trade does not contribute to the avg_price calculation but only realizes the profit/loss of the position.
                Therefore, we need to make sure that the position has flipped or closed before calculating the avg_price.
                From then on, we can also calculate stop-loss/take-profit correctly.
        2. To ensure #1 is always true and the position can always be closed, 
            the actual total traded volume available is ignored when closing position.
            i.e. the traded volume for closing position might be larger than the actual traded volume in 'volume' column,
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
        6. conceptually that we are at the end of bar N, assume only one trade per bar. order of precedence (highest to lowest):
            1. immediately triggered stop order (SL/TP, 'close' price already breaches stop price)
            2. time window close order (max holding period reached)
            3. market order (opened order where order_price >= close)
            4. at bar N+1, limit order filled first, only when limit order price is better than stop price
            5. non-immediately triggered stop order (SL/TP, high/low breaches stop price during the bar)
        ''' 
        def _calculate_stop_price():
            '''Calculates stop price for stop-loss, take-profit and time_window at the same time,
            only triggered stop prices are shown.
            NOTE: time_window close orders also reuse the stop_price column to avoid duplicating cleanup logic.
            Keeps the first triggered stop order for each position streak,
            updates the trade_size and trade_price accordingly,
            and clean up orders and trades after stop orders.
            '''
            end_position_side = np.sign(self['position'])
            start_position_side = end_position_side.shift(1, fill_value=0)
            prev_close = self['close'].shift(1)
            opened_order_side = self['signal'].shift(1)
            opened_order_price = self['order_price'].shift(1)
            long_order = (opened_order_side == 1)
            short_order = (opened_order_side == -1)
            market_order_trade_condition = (
                (long_order & (prev_close <= opened_order_price)) |
                (short_order & (prev_close >= opened_order_price))
            )
            limit_order_trade_condition = (
                (long_order & (opened_order_price >= self['low'])) |
                (short_order & (opened_order_price <= self['high']))
            )
            
            for tp_or_sl, sign in [(stop_loss, -1), (take_profit, 1)]:
                if tp_or_sl is None:
                    continue
                stop_price = self['avg_price'] * (1 + end_position_side * tp_or_sl * sign)
                # think of it as stop order was placed at the end of bar N, and now it's at the beginning of bar N+1
                opened_stop_order_price = stop_price.shift(1)
                positive_sign = (start_position_side * sign == 1)
                negative_sign = (start_position_side * sign == -1)

                # stop orders that are triggered immediately
                # NOTE: here prev_close is used instead of self['open'] to trigger market orders
                # because market_order uses prev_close (for convenience), 
                # and since stop market order has higher priority than market order, it should be also use prev_close for consistency
                stop_market_order_triggered_condition_immediate = (
                    (negative_sign & (prev_close <= opened_stop_order_price)) |
                    (positive_sign & (prev_close >= opened_stop_order_price))    
                )
                
                stop_market_order_triggered_condition = (
                    (negative_sign & (self['low'] <= opened_stop_order_price)) |
                    (positive_sign & (self['high'] >= opened_stop_order_price))
                )
                
                # Concept:
                # - An opened order is "market" if it is already aggressive vs prev_close:
                #     long: opened_order_price >= prev_close
                #     short: opened_order_price <= prev_close
                #   Otherwise it is a limit order.
                # - Here we compare only the relative level of a LIMIT order vs stop level to decide
                #   which can be reached first on the same side of the book.
                #   * downside case (negative_sign + long_order): higher price is reached first while moving down
                #       -> limit-first if opened_order_price >= opened_stop_order_price
                #   * upside case (positive_sign + short_order): lower price is reached first while moving up
                #       -> limit-first if opened_order_price <= opened_stop_order_price
                # Assumption behind this intuition: no-gap path (open ~= prev_close). With gaps, precedence
                # rules remain deterministic, but path intuition is weaker.
                is_limit_order_filled_first = (
                    limit_order_trade_condition
                    & (
                       negative_sign & long_order & (opened_order_price >= opened_stop_order_price) |
                       positive_sign & short_order & (opened_order_price <= opened_stop_order_price)
                    )
                )
                
                stop_market_order_trade_condition = np.where(
                    stop_market_order_triggered_condition_immediate,
                    True,
                    np.where(
                        stop_market_order_triggered_condition & (~market_order_trade_condition) & (~is_limit_order_filled_first),
                        True,
                        False
                    )
                )
                
                # only keep those triggered stop prices
                self['stop_price'] = np.where(
                    # needs df['stop_price'].isna() so that the first stop price is kept, i.e. stop loss won't be overridden by take profit
                    stop_market_order_trade_condition & self['stop_price'].isna(),  
                    opened_stop_order_price,
                    self['stop_price']
                )
            self['stop_price'] = self['stop_price'].shift(-1)  # shift back stop_price from trade row to order row

            # NOTE: time_window close orders reuse the stop_price column,
            # written after SL/TP so that SL/TP have higher priority (limitation #6).
            # stop_price is already on the order row after shift(-1) above.
            if time_window:
                has_position = (self['position'] != 0)
                global_bar_count = has_position.cumsum()
                streak_start_bar_count = global_bar_count.shift(1, fill_value=0).where(self['_position_change']).ffill().fillna(0)
                bar_count = global_bar_count - streak_start_bar_count
                # _time_window is on the order row: place a market close order at end of this bar
                self['_time_window'] = (bar_count == time_window) & has_position
                self['stop_price'] = np.where(
                    self['_time_window'] & self['stop_price'].isna(),
                    self['close'],
                    self['stop_price']
                )


        if take_profit is not None:
            assert take_profit > 0, "'take_profit' must be positive"
        if stop_loss is not None:
            stop_loss = abs(stop_loss)
            assert 1 > stop_loss > 0, "'stop_loss' must be between 0 and 1"
        if time_window is not None:
            assert isinstance(time_window, int) and time_window > 0, "'time_window' must be a positive integer"
        
        # Step 1: calculate position
        # NOTE: some trade_size values were set to 0s (placeholders) in open_position() to indicate places to close the position for long_only/short_only strategies
        # so that "_position_change" below can be computed directly
        trade_side = np.sign(self['trade_size'])

        # position change = position closed or flipped
        self['_position_change'] = trade_side.ffill().diff().ne(0) & trade_side.notna()
        
        trade_size_filled = self['trade_size'].fillna(0)
        global_cumsum = trade_size_filled.cumsum()
        # At each streak boundary, capture the cumsum value just before the new streak starts
        streak_start_cumsum = global_cumsum.shift(1, fill_value=0).where(self['_position_change']).ffill().fillna(0)
        self['position'] = global_cumsum - streak_start_cumsum

        # NOTE: groupby version of the above, keeping it for readability
        # position streak also includes opposite signals if not traded, e.g. +1, +1, +1, -1 (no trade), -1 (no trade), ...
        # self['_position_streak'] = self['_position_change'].cumsum()
        # self['position'] = self.groupby('_position_streak')['trade_size'].transform(
        #     lambda x: x.cumsum().ffill()
        # ).fillna(0)
        
        
        # Step 2: calculate avg_price
        cost = (self['trade_price'].fillna(0) * self['trade_size'].fillna(0))
        global_cost_cumsum = cost.cumsum()
        streak_start_cost_cumsum = global_cost_cumsum.shift(1, fill_value=0).where(self['_position_change']).ffill().fillna(0)
        self['_agg_costs'] = global_cost_cumsum - streak_start_cost_cumsum
        self['avg_price'] = np.where(self['position'] != 0, self['_agg_costs'] / self['position'], np.nan)
        self['avg_price'] = self['avg_price'].ffill()
        
        
        # Step 3: calculate stop_price and handle close conditions (SL/TP/time_window)
        self['stop_price'] = np.nan
        has_close_condition = bool(take_profit or stop_loss or time_window)
        if has_close_condition:
            _calculate_stop_price()
            end_position_side = np.sign(self['position'])
            # NOTE: applying the same logic as _first_trade in open_position()
            stop_price_notna = self['stop_price'].notna()
            self['_stop_side'] = np.where(
                stop_price_notna,
                end_position_side,
                np.where(self['_position_change'], 0, np.nan)
            )
            stop_side_ffill = self['_stop_side'].ffill()
            self['_first_stop_order'] = (
                stop_side_ffill.diff().ne(0)
                & stop_price_notna  # filter out 0s in _stop_side
            )

            # clean up stop_price, only the first one in each streak is left
            self['stop_price'] = np.where(self['_first_stop_order'], self['stop_price'], np.nan)

            # determine close reason flags (all on the order row, only at _first_stop_order rows)
            if time_window:
                # narrow _time_window to only the first stop orders that were from time_window
                self['_time_window'] = self['_first_stop_order'] & self['_time_window']
            if take_profit or stop_loss:
                not_time_window = ~self['_time_window'] if time_window else True
                price_diff_check = end_position_side * (self['avg_price'] - self['stop_price'])
                self['_stop_loss'] = self['_first_stop_order'] & not_time_window & (price_diff_check > 0)
                self['_take_profit'] = self['_first_stop_order'] & not_time_window & (price_diff_check < 0)

            # update trades created by stop/time_window orders
            offset_size = self['position'].shift(1, fill_value=0) * (-1)
            first_stop_trade = self['_first_stop_order'].shift(1, fill_value=False)
            self['trade_size'] = np.where(first_stop_trade, offset_size, self['trade_size'])
            self['trade_price'] = np.where(
                first_stop_trade,
                self['stop_price'].shift(1),  # self['stop_price'].shift(1) * (1 - start_position_side * slippage),
                self['trade_price']
            )

            # clean up order, trades and 'position' after stop/time_window orders
            first_stop_order_forwards_mask = stop_side_ffill.fillna(0).ne(0)
            order_mask = first_stop_order_forwards_mask & (self['signal'] == np.sign(stop_side_ffill)) & self['stop_price'].isna()
            self['order_size'] = np.where(order_mask, np.nan, self['order_size'])
            self['order_price'] = np.where(order_mask, np.nan, self['order_price'])
            position_mask = first_stop_order_forwards_mask & (~self['_first_stop_order'])
            self['position'] = np.where(position_mask, 0.0, self['position'])
            trade_mask = position_mask & (~first_stop_trade)
            self['trade_size'] = np.where(trade_mask, np.nan, self['trade_size'])
            self['trade_price'] = np.where(trade_mask, np.nan, self['trade_price'])

            
        # Step 4: clean up avg_price, position and trades with or without stop orders
        self['avg_price'] = np.where(self['position'] == 0, np.nan, self['avg_price'])
        
        offset_order_size = self['position'] * (-1)
        offset_trade_size = offset_order_size.shift(1, fill_value=0)
        # override trade_size and order_size with the offset sizes
        self['trade_size'] = np.where(
            self['_position_change'] & (offset_trade_size != 0) & self['stop_price'].shift(1).isna(), 
            offset_trade_size + self['trade_size'], 
            self['trade_size']
        )
        self['order_size'] = np.where(
            (np.sign(offset_order_size) == self['signal']) & (offset_order_size != 0),
            offset_order_size + self['order_size'], 
            self['order_size']
        )
        if self['stop_price'].isna().all():
            self.drop(columns=['stop_price'], inplace=True)
            
        return self