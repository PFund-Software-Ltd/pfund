# pyright: reportAssignmentType=false, reportUnknownMemberType=false
from typing import Any, Literal, Self

import polars as pl

from pfund._backtest.narwhals_mixin import NarwhalsMixin
from pfund.enums import BacktestMode


# TODO: test on engine="gpu"
# pl.Config.set_engine_affinity(engine="streaming")


# TODO: maybe create a subclass like SafeFrame(pd.DataFrame) to prevent users from peeking into the future?
# e.g. df['close'] = df['close'].shift(-1) should not be allowed
class BacktestDataFrame(NarwhalsMixin, pl.DataFrame):
    def __init__(self, *args: Any, backtest_mode: BacktestMode, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._backtest_mode: BacktestMode = backtest_mode
        if self._backtest_mode not in [BacktestMode.VECTORIZED, BacktestMode.HYBRID]:
            raise ValueError(f"Invalid backtest mode: {self._backtest_mode}")
        self._open_position_inputs = {}
        self._close_position_inputs = {}

    def create_signal(
        self,
        buy_condition: pl.Series | None=None,
        sell_condition: pl.Series | None=None,
        signal: pl.Series | None=None,
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
                raise ValueError("Either buy or sell must be provided")
            if buy_condition is not None and sell_condition is not None:
                overlaps = buy_condition & sell_condition
                if overlaps.any():
                    raise ValueError(
                        "Overlapping buy and sell condition detected.\n" + 
                        "Please make sure that buy and sell conditions are mutually exclusive."
                    )
                signal_series = (
                    pl.DataFrame({'_buy': buy_condition, '_sell': sell_condition})
                    .select(
                        pl
                        .when(pl.col('_buy')).then(1)
                        .when(pl.col('_sell')).then(-1)
                        .otherwise(None)
                        .alias('signal')
                    )
                    .get_column('signal')
                )
            else:
                cond = buy_condition if buy_condition is not None else sell_condition
                value = 1 if buy_condition is not None else -1
                signal_series = (
                    pl.DataFrame({'_cond': cond})
                    .select(
                        pl
                        .when(pl.col('_cond')).then(value)
                        .otherwise(None)
                        .alias('signal')
                    )
                    .get_column('signal')
                )
        else:
            assert signal.drop_nulls().unique().is_in([1, -1]).all(), "'signal' must only contain 1, -1, null"
            signal_series = signal.alias('signal')

        # _signal_change: True where the forward-filled signal differs from its previous value
        signal_ffill = signal_series.forward_fill()
        signal_change = (
            signal_ffill.ne(signal_ffill.shift(1)).fill_null(True)
            & signal_ffill.is_not_null()
        )

        df = self.with_columns([
            signal_series,
            signal_change.alias('_signal_change'),
        ])

        if first_only:
            df = df.with_columns(
                pl.when(pl.col('_signal_change'))
                .then(pl.col('signal'))
                .otherwise(None)
                .alias('signal')
            )

        return self.__class__(df, backtest_mode=self._backtest_mode)

    def open_position(
        self,
        order_price: pl.Series | None=None,
        order_quantity: pl.Series | int | float=1,
        first_only: bool=False,
        long_only: bool=False,
        short_only: bool=False,
        fill_price: Literal['open', 'close']='close',
    ) -> Self:
        '''
        Opens positions in a vectorized manner.
        Conceptually, this function places orders at the end of bar/candlestick N.
        For example, for a buy order:
        - If the order price >= close price of bar N, it is a market order,
            by assuming that the close price is the current best price; otherwise it is a limit order.
        Then the orders are opened at the beginning of bar N+1,
        and filled during bar N+1 if high >= order price >= low.
        If bar N+1 gaps through the limit (order price is outside [low, high]),
        the order is treated as a marketable gap-through and fills at N+1 open.
        Opened orders are considered as cancelled at the end of bar N+1 if not filled.
        Args:
            order_price: price to place the order.
                If None, use 'close' price (market order).
            order_quantity: quantity to place the order.
            first_only: first trade only, do not trade after the first trade until signal changes
            long_only: all order_size in signal=-1 will be set to 0, meaning the order_size will be determined in close_position()
                useful for long-only strategy with signal=-1 to close the position
            short_only: all order_size in signal=1 will be set to 0, meaning the order_size will be determined in close_position()
                useful for short-only strategy with signal=1 to close the position
            fill_price: fill price for market orders.
                An order is placed at bar N's close. 'close' fills at bar N's close price,
                'open' fills at bar N+1's open price.
                Applies to regular market orders (aggressive vs close).
                Gap-through limits always fill at bar N+1 open.
                In-range limit orders fill at their limit price.
                Default is 'close', which avoids gap exposure (e.g. overnight gaps on daily bars).
        '''
        assert 'signal' in self.columns, "No 'signal' column is found, please use create_signal() first"
        assert not (long_only and short_only), "Cannot be long_only and short_only at the same time"
        assert fill_price in ('open', 'close'), "'fill_price' must be 'open' or 'close'"

        # HYBRID mode will delay everything to backtest_loop(), store the inputs here first
        if self._backtest_mode == BacktestMode.HYBRID:
            self._open_position_inputs = {
                'order_price': order_price,
                'order_quantity': order_quantity,
                'first_only': first_only,
                'long_only': long_only,
                'short_only': short_only,
                'fill_price': fill_price,
            }
            # Keep stored hybrid inputs on the same object; returning a new
            # BacktestDataFrame would reset these attributes in __init__.
            return self

        # Validate that opposite signals exist for long_only/short_only.
        # Without opposite signals, positions can only be closed by SL/TP,
        # but re-entry after SL/TP close is not possible in vectorized mode (limitation #3).
        signal_values = self.get_column('signal').drop_nulls().unique()
        if long_only and not signal_values.is_in([-1]).any():
            raise ValueError(
                'long_only=True with no sell signals (-1) is not supported in vectorized mode. ' +
                'Without opposite signals, positions can only be closed by SL/TP, ' +
                'but re-entry is not possible after that. ' +
                'Please use backtest mode="hybrid" instead.'
            )
        if short_only and not signal_values.is_in([1]).any():
            raise ValueError(
                'short_only=True with no buy signals (1) is not supported in vectorized mode. ' +
                'Without opposite signals, positions can only be closed by SL/TP, ' +
                'but re-entry is not possible after that. ' +
                'Please use backtest mode="hybrid" instead.'
            )

        signal = self.get_column('signal')

        # 1. create orders
        if order_price is None:
            order_price = self.get_column('close')
        else:
            assert ( (order_price > 0) | order_price.is_null() ).all(), "'order_price' must be positive or null"
        if isinstance(order_quantity, pl.Series):
            assert ( (order_quantity > 0) | order_quantity.is_null() ).all(), "'order_quantity' values must be positive or null"
        else:
            assert order_quantity > 0, "'order_quantity' must be positive"
        df = self.with_columns(
            order_price=signal.abs() * order_price,
            order_size=signal * order_quantity,
        )

        # 2. place orders
        # shift 'order_price' and 'order_size' to the next bar and act as opened limit orders in the same row
        # NOTE: order placed at the end of the previous bar = opened order at the beginning of the next bar
        opened_order_price = pl.col('order_price').shift(1)
        opened_order_size = pl.col('order_size').shift(1)
        opened_order_side = pl.col('signal').shift(1)

        # 3. fill orders
        # trade_price = min(trade_price, open) if buy, max(trade_price, open) if sell
        prev_close = pl.col('close').shift(1)
        long_order = opened_order_side == 1
        short_order = opened_order_side == -1
        # NOTE: here prev_close is used instead of df['open'] to trigger market orders
        # because it's convenient to place market orders by setting order_price=df['close']
        # Regular market order: order price aggressive vs prev_close
        regular_market_condition = (
            (long_order & (prev_close <= opened_order_price)) |
            (short_order & (prev_close >= opened_order_price))
        )
        # Gap-through: open crosses through the limit price → always fills at open
        gap_through_market_condition = (
            (long_order & ~regular_market_condition & (pl.col('open') < opened_order_price)) |
            (short_order & ~regular_market_condition & (pl.col('open') > opened_order_price))
        )
        market_order_trade_condition = regular_market_condition | gap_through_market_condition
        # Limit order: one-sided check suffices since gap-through is absorbed into market
        limit_order_trade_condition = (
            (long_order & (opened_order_price >= pl.col('low'))) |
            (short_order & (opened_order_price <= pl.col('high')))
        )
        trade_condition = market_order_trade_condition | limit_order_trade_condition
        # Fill prices: gap-through always at open; regular market follows fill_price policy
        regular_market_fill = pl.col('open') if fill_price == 'open' else prev_close
        df = df.with_columns(
            trade_price=(
                pl.when(gap_through_market_condition).then(pl.col('open'))
                .when(regular_market_condition).then(regular_market_fill)
                .when(limit_order_trade_condition).then(opened_order_price)
                .otherwise(None)
            ),
            # NOTE: ignore the actual traded 'volume', just use the order size directly
            trade_size=pl.when(trade_condition).then(opened_order_size).otherwise(None),
        )

        # this version takes the actual traded 'volume' into account
        # trade_size=(
        #     pl.when(trade_condition & (opened_order_size.abs() > pl.col('volume')))
        #     .then(pl.col('volume') * opened_order_side)
        #     .when(trade_condition)
        #     .then(opened_order_size)
        #     .otherwise(None)
        # ),

        if first_only or long_only or short_only:
            if first_only:
                opposite_side = None
                filtered_orders = pl.lit(True)
            else:  # long_only or short_only with first_only=False:
                opposite_side = -1 if long_only else 1
                filtered_orders = pl.col('signal') == opposite_side
            filtered_trades = filtered_orders.shift(1).fill_null(False if opposite_side is not None else True)

            # To determine the first trade of each trade streak, create trade streaks using trade sides (+1/-1/0)
            # trade side = 0 means no trade but signal changed, used to separate two trade streaks of the same sign but no trades in the middle,
            # e.g. signal streak: +1, +1 (trade), +1, -1 (no trade, 0 added to trade side), +1, +1 (trade)
            # without trade_side=0, this example would be treated as one trade streak +1, +1, -1, +1, +1
            trade_price_notna = pl.col('trade_price').is_not_null()
            df = df.with_columns(
                pl.when(trade_price_notna)
                .then(pl.col('trade_size').sign())
                .when(pl.col('_signal_change').shift(1).fill_null(False))
                .then(0)
                .otherwise(None)
                .alias('_trade_side')
            )
            trade_side_ffill = pl.col('_trade_side').forward_fill()
            # first trade of the filtered trades
            df = df.with_columns(
                (
                    trade_side_ffill.diff().ne(0).fill_null(True)
                    & trade_price_notna  # filter out 0s in _trade_side
                    & filtered_trades
                ).alias('_first_trade')
            )

            # clean up orders and trades after the first trade
            order_mask = (
                trade_side_ffill.fill_null(0).ne(0)
                & ~pl.col('_signal_change')
                & filtered_orders
            )
            trade_mask = pl.col('_first_trade') | ~filtered_trades
            df = df.with_columns(
                order_size=pl.when(order_mask).then(None).otherwise(pl.col('order_size')),
                order_price=pl.when(order_mask).then(None).otherwise(pl.col('order_price')),
                trade_size=pl.when(trade_mask).then(pl.col('trade_size')).otherwise(None),
                trade_price=pl.when(trade_mask).then(pl.col('trade_price')).otherwise(None),
            )

            if long_only or short_only:
                if opposite_side is None:
                    opposite_side = -1 if long_only else 1
                    filtered_orders = pl.col('signal') == opposite_side
                    filtered_trades = filtered_orders.shift(1).fill_null(False)
                # By setting the order size=0, it means the size will be determined in close_position()
                # as the position offset size, i.e. the position will not be flipped when order_size=0
                df = df.with_columns(
                    order_size=(
                        pl.when(filtered_orders & pl.col('order_price').is_not_null())
                        .then(0)
                        .otherwise(pl.col('order_size'))
                    ),
                    trade_size=(
                        pl.when(filtered_trades & pl.col('trade_price').is_not_null())
                        .then(0)
                        .otherwise(pl.col('trade_size'))
                    ),
                )

        return self.__class__(df, backtest_mode=self._backtest_mode)

    def close_position(
        self,
        take_profit: float | None=None,
        stop_loss: float | None=None,
        time_window: int | None=None,
        fill_price: Literal['open', 'close']='close',
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
        2. Volume is ignored: trades fill at the full order size regardless of the bar's actual traded volume.
        3. position cannot be re-entered after being closed by stop-loss/take-profit in the same trade streak
            This is a huge limitation for long-only/short-only strategies with only +1s/only -1s signals.
            For example, if the strategy is long-only and depends on stop-loss to close the position,
            a for loop is needed since #1 no longer holds in this case.
            However, if this long-only strategy has prepared -1 signals in advance,
            instead of relying on stop-loss to close the position,
            then #1 still applies.
        4. stop_loss/take_profit only supports stop market orders, because the exact price movement after stop_loss/take_profit is unknown,
            placing limit orders might not get filled.
            Immediately triggered stops have two cases:
            - regular immediate: prev_close breaches stop price, respects fill_price
            - gap-through immediate: stop price is outside [low, high], fills at next open
            Non-immediately triggered stops (high/low breaches stop price during the bar) fill at the stop trigger price.
        5. assumes stop loss is always triggered before take profit if the order of happening cannot be determined.
            - in long position, check 'low' first; in short position, check 'high' first
        6. conceptually that we are at the end of bar N, assume only one trade per bar. order of precedence (highest to lowest):
            1. immediately triggered stop order (SL/TP, 'close' price already breaches stop price)
            2. time window close order (max holding period reached)
            3. market order (opened order where order_price >= close)
            4. at bar N+1, limit order filled first, only when limit order price is better than stop price
            5. non-immediately triggered stop order (SL/TP, high/low breaches stop price during the bar)

        Args:
            take_profit: take profit percentage (e.g. 0.1 = 10%).
            stop_loss: stop loss percentage between 0 and 1 (e.g. 0.05 = 5%).
            time_window: max number of bars to hold a position before auto-closing.
            fill_price: fill price for market close orders (immediately triggered SL/TP and time_window).
                These trigger at bar N's close. 'close' fills at bar N's close price,
                'open' fills at bar N+1's open price.
                Gap-through immediate SL/TP (stop outside [low, high]) always fill at bar N+1 open.
                Non-immediately triggered SL/TP (high/low breach during bar) always fill at their trigger price.
                Default is 'close', which avoids gap exposure (e.g. overnight gaps on daily bars).
        '''
        def _calculate_stop_price(_df: pl.DataFrame) -> pl.DataFrame:
            '''Calculates stop price for stop-loss, take-profit and time_window at the same time,
            only triggered stop prices are shown.
            NOTE: time_window close orders also reuse the stop_price column to avoid duplicating cleanup logic.
            Keeps the first triggered stop order for each position streak,
            updates the trade_size and trade_price accordingly,
            and clean up orders and trades after stop orders.
            '''
            end_position_side = pl.col('position').sign()
            start_position_side = end_position_side.shift(1).fill_null(0)
            prev_close = pl.col('close').shift(1)
            opened_order_side = pl.col('signal').shift(1)
            opened_order_price = pl.col('order_price').shift(1)
            long_order = (opened_order_side == 1).fill_null(False)
            short_order = (opened_order_side == -1).fill_null(False)
            # Market order: order price aggressive vs prev_close, or gap-through (price outside [low, high])
            market_order_trade_condition = (
                (long_order & ((prev_close <= opened_order_price) | (opened_order_price > pl.col('high')))) |
                (short_order & ((prev_close >= opened_order_price) | (opened_order_price < pl.col('low'))))
            ).fill_null(False)
            # Limit order: one-sided check suffices since gap-through is absorbed into market.
            # NOTE: this is only used for limit_before_stop priority, not for fill price.
            limit_order_trade_condition = (
                (long_order & (opened_order_price >= pl.col('low'))) |
                (short_order & (opened_order_price <= pl.col('high')))
            ).fill_null(False)

            _df = _df.with_columns(
                _immediate_stop=pl.lit(False),
                _gap_through_stop=pl.lit(False),
            )
            for tp_or_sl, sign in [(stop_loss, -1), (take_profit, 1)]:
                if tp_or_sl is None:
                    continue
                stop_price_computed = pl.col('avg_price') * (1 + end_position_side * tp_or_sl * sign)
                opened_stop_order_price = stop_price_computed.shift(1)
                positive_sign = (start_position_side * sign).eq(1)
                negative_sign = (start_position_side * sign).eq(-1)

                # Regular immediate: prev_close already breaches stop price
                regular_immediate_stop = (
                    (negative_sign & (prev_close <= opened_stop_order_price)) |
                    (positive_sign & (prev_close >= opened_stop_order_price))
                )
                # Gap-through: open crosses through the stop price — always fills at open
                gap_through_stop = (
                    (negative_sign & ~regular_immediate_stop & (pl.col('open') < opened_stop_order_price)) |
                    (positive_sign & ~regular_immediate_stop & (pl.col('open') > opened_stop_order_price))
                )
                stop_market_order_triggered_condition_immediate = regular_immediate_stop | gap_through_stop

                # Non-immediate: one-sided check suffices since gap-through is absorbed into immediate
                stop_market_order_triggered_condition = (
                    (negative_sign & (pl.col('low') <= opened_stop_order_price)) |
                    (positive_sign & (pl.col('high') >= opened_stop_order_price))
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
                       (negative_sign & long_order & (opened_order_price >= opened_stop_order_price)) |
                       (positive_sign & short_order & (opened_order_price <= opened_stop_order_price))
                    )
                )

                stop_market_order_trade_condition = (
                    stop_market_order_triggered_condition_immediate
                    | (stop_market_order_triggered_condition & ~market_order_trade_condition & ~is_limit_order_filled_first)
                )

                # only keep those triggered stop prices
                # needs stop_price.is_null() so that the first stop price is kept, i.e. stop loss won't be overridden by take profit
                should_set_stop = stop_market_order_trade_condition & pl.col('stop_price').is_null()
                _df = _df.with_columns(
                    pl.when(should_set_stop).then(opened_stop_order_price).otherwise(pl.col('stop_price')).alias('stop_price'),
                    pl.when(should_set_stop).then(stop_market_order_triggered_condition_immediate).otherwise(pl.col('_immediate_stop')).alias('_immediate_stop'),
                    pl.when(should_set_stop).then(gap_through_stop).otherwise(pl.col('_gap_through_stop')).alias('_gap_through_stop'),
                )

            _df = _df.with_columns(
                stop_price=pl.col('stop_price').shift(-1),  # shift back stop_price from trade row to order row
                _immediate_stop=pl.col('_immediate_stop').shift(-1).fill_null(False),
                _gap_through_stop=pl.col('_gap_through_stop').shift(-1).fill_null(False),
            )

            # NOTE: time_window close orders reuse the stop_price column,
            # written after SL/TP so that SL/TP have higher priority (limitation #6).
            # stop_price is already on the order row after shift(-1) above.
            if time_window:
                has_position = pl.col('position').ne(0)
                global_bar_count = has_position.cum_sum()
                streak_start_bar_count = (
                    pl.when(pl.col('_position_change'))
                    .then(global_bar_count.shift(1).fill_null(0))
                    .otherwise(None)
                    .forward_fill()
                    .fill_null(0)
                )
                bar_count = global_bar_count - streak_start_bar_count
                # _time_window is on the order row: place a market close order at end of this bar
                _df = _df.with_columns(
                    _time_window=(bar_count.eq(time_window) & has_position)
                )
                # Use close price as trigger price for time_window:
                # 1. Conceptually correct — close price is the reference for market order detection
                # 2. The actual fill price (trade_price) is handled separately via fill_price logic downstream
                _df = _df.with_columns(
                    pl.when(pl.col('_time_window') & pl.col('stop_price').is_null())
                    .then(pl.col('close'))
                    .otherwise(pl.col('stop_price'))
                    .alias('stop_price')
                )

            return _df
        
        
        if take_profit is not None:
            assert take_profit > 0, "'take_profit' must be positive"
        if stop_loss is not None:
            stop_loss = abs(stop_loss)
            assert 1 > stop_loss > 0, "'stop_loss' must be between 0 and 1"
        if time_window is not None:
            assert isinstance(time_window, int) and time_window > 0, "'time_window' must be a positive integer"
        assert fill_price in ('open', 'close'), "'fill_price' must be 'open' or 'close'"
        
        
        # HYBRID mode will delay everything to backtest_loop(), store the inputs here first
        if self._backtest_mode == BacktestMode.HYBRID:
            self._close_position_inputs = {
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'time_window': time_window,
                'fill_price': fill_price,
            }
            # Keep stored hybrid inputs on the same object; returning a new
            # BacktestDataFrame would reset these attributes in __init__.
            return self


        # Step 1: calculate position
        # NOTE: some trade_size values were set to 0s (placeholders) in open_position() to indicate places to close the position for long_only/short_only strategies
        # so that "_position_change" below can be computed directly
        trade_side = pl.col('trade_size').sign()

        # position change = position closed or flipped
        df = self.with_columns(
            _position_change=(
                trade_side.forward_fill().diff().ne(0).fill_null(True)
                & trade_side.is_not_null()
            )
        )

        trade_size_filled = pl.col('trade_size').fill_null(0)
        global_cumsum = trade_size_filled.cum_sum()
        # At each streak boundary, capture the cumsum value just before the new streak starts
        streak_start_cumsum = (
            pl.when(pl.col('_position_change'))
            .then(global_cumsum.shift(1).fill_null(0))
            .otherwise(None)
            .forward_fill()
            .fill_null(0)
        )
        df = df.with_columns(
            position=(global_cumsum - streak_start_cumsum)
        )


        # Step 2: calculate avg_price
        cost = pl.col('trade_price').fill_null(0) * pl.col('trade_size').fill_null(0)
        global_cost_cumsum = cost.cum_sum()
        streak_start_cost_cumsum = (
            pl.when(pl.col('_position_change'))
            .then(global_cost_cumsum.shift(1).fill_null(0))
            .otherwise(None)
            .forward_fill()
            .fill_null(0)
        )
        df = df.with_columns(
            _agg_costs=(global_cost_cumsum - streak_start_cost_cumsum)
        )
        df = df.with_columns(
            avg_price=(
                pl.when(pl.col('position').ne(0))
                .then(pl.col('_agg_costs') / pl.col('position'))
                .otherwise(None)
            )
        )
        df = df.with_columns(
            avg_price=pl.col('avg_price').forward_fill()
        )
        

        # Step 3: calculate stop_price and handle close conditions (SL/TP/time_window)
        df = df.with_columns(stop_price=pl.lit(None).cast(pl.Float64))
        has_close_condition = bool(take_profit or stop_loss or time_window)
        if has_close_condition:
            df = _calculate_stop_price(df)
            end_position_side = pl.col('position').sign()
            stop_price_notna = pl.col('stop_price').is_not_null()
            # NOTE: applying the same logic as _first_trade in open_position()
            df = df.with_columns(
                pl.when(stop_price_notna)
                .then(end_position_side)
                .when(pl.col('_position_change'))
                .then(0)
                .otherwise(None)
                .alias('_stop_side')
            )
            stop_side_ffill = pl.col('_stop_side').forward_fill()
            df = df.with_columns(
                (
                    stop_side_ffill.diff().ne(0).fill_null(True)
                    & stop_price_notna  # filter out 0s in _stop_side
                ).alias('_first_stop_order')
            )

            # clean up stop_price, only the first one in each streak is left
            df = df.with_columns(
                pl.when(pl.col('_first_stop_order'))
                .then(pl.col('stop_price'))
                .otherwise(None)
                .alias('stop_price')
            )

            # determine close reason flags (all on the order row, only at _first_stop_order rows)
            if time_window:
                # narrow _time_window to only the first stop orders that were from time_window
                df = df.with_columns(
                    _time_window=(pl.col('_first_stop_order') & pl.col('_time_window'))
                )
            if take_profit or stop_loss:
                not_time_window = ~pl.col('_time_window') if time_window else pl.lit(True)
                price_diff_check = pl.col('position').sign() * (pl.col('avg_price') - pl.col('stop_price'))
                df = df.with_columns(
                    _stop_loss=(pl.col('_first_stop_order') & not_time_window & (price_diff_check > 0)),
                    _take_profit=(pl.col('_first_stop_order') & not_time_window & (price_diff_check < 0)),
                )

            # update trades created by stop/time_window orders
            offset_size = pl.col('position').shift(1).fill_null(0) * (-1)
            first_stop_trade = pl.col('_first_stop_order').shift(1).fill_null(False)
            stop_fill = pl.col('stop_price').shift(1)
            # Gap-through stops always fill at open (prev_close didn't breach the stop)
            is_gap_stop_trade = pl.col('_gap_through_stop').shift(1).fill_null(False)
            stop_fill = (
                pl.when(is_gap_stop_trade).then(pl.col('open'))
                .otherwise(stop_fill)
            )
            # Regular immediate stops and time_window close orders follow fill_price policy
            is_market_stop_trade = pl.col('_immediate_stop').shift(1).fill_null(False) & ~is_gap_stop_trade
            if time_window:
                is_market_stop_trade = is_market_stop_trade | pl.col('_time_window').shift(1).fill_null(False)
            if fill_price == 'open':
                stop_fill = (
                    pl.when(is_market_stop_trade).then(pl.col('open'))
                    .otherwise(stop_fill)
                )
            else:  # fill_price == 'close'
                stop_fill = (
                    pl.when(is_market_stop_trade).then(pl.col('close').shift(1))
                    .otherwise(stop_fill)
                )
            df = df.with_columns(
                trade_size=pl.when(first_stop_trade).then(offset_size).otherwise(pl.col('trade_size')),
                trade_price=pl.when(first_stop_trade).then(stop_fill).otherwise(pl.col('trade_price')),
            )

            # clean up order, trades and 'position' after stop/time_window orders
            first_stop_order_forwards_mask = stop_side_ffill.fill_null(0).ne(0)
            order_mask = first_stop_order_forwards_mask & pl.col('signal').eq(stop_side_ffill.sign()) & pl.col('stop_price').is_null()
            position_mask = first_stop_order_forwards_mask & ~pl.col('_first_stop_order')
            trade_mask = position_mask & ~first_stop_trade
            df = df.with_columns(
                order_size=pl.when(order_mask).then(None).otherwise(pl.col('order_size')),
                order_price=pl.when(order_mask).then(None).otherwise(pl.col('order_price')),
                position=pl.when(position_mask).then(0.0).otherwise(pl.col('position')),
                trade_size=pl.when(trade_mask).then(None).otherwise(pl.col('trade_size')),
                trade_price=pl.when(trade_mask).then(None).otherwise(pl.col('trade_price')),
            )


        # Step 4: clean up avg_price, position and trades with or without stop orders
        df = df.with_columns(
            avg_price=pl.when(pl.col('position').eq(0)).then(None).otherwise(pl.col('avg_price')),
        )

        offset_order_size = pl.col('position') * (-1)
        offset_trade_size = (pl.col('position') * (-1)).shift(1).fill_null(0)
        # override trade_size and order_size with the offset sizes
        df = df.with_columns(
            trade_size=(
                pl.when(
                    pl.col('_position_change') & offset_trade_size.ne(0) & pl.col('stop_price').shift(1).is_null()
                )
                .then(offset_trade_size + pl.col('trade_size'))
                .otherwise(pl.col('trade_size'))
            ),
            order_size=(
                pl.when(
                    offset_order_size.sign().eq(pl.col('signal')) & offset_order_size.ne(0)
                )
                .then(offset_order_size + pl.col('order_size'))
                .otherwise(pl.col('order_size'))
            ),
        )
        if df.get_column('stop_price').is_null().all():
            df = df.drop('stop_price')

        # Clean up 0-size no-op trades/orders (e.g. long_only close signals when no position exists)
        df = df.with_columns(
            pl.when(pl.col('trade_size').eq(0))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col('trade_price'))
            .alias('trade_price'),
            pl.when(pl.col('trade_size').eq(0))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col('trade_size'))
            .alias('trade_size'),
        )
        df = df.with_columns(
            pl.when(pl.col('order_size').eq(0))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col('order_price'))
            .alias('order_price'),
            pl.when(pl.col('order_size').eq(0))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col('order_size'))
            .alias('order_size'),
        )

        return self.__class__(df, backtest_mode=self._backtest_mode)
