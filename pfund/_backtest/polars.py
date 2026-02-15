# pyright: reportAssignmentType=false
from typing import Self

import polars as pl


# TODO: test on engine="gpu"
# pl.Config.set_engine_affinity(engine="streaming")


# TODO: maybe create a subclass like SafeFrame(pd.DataFrame) to prevent users from peeking into the future?
# e.g. df['close'] = df['close'].shift(-1) should not be allowed
class BacktestDataFrame(pl.DataFrame):
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

        return self.__class__(df)

    def open_position(
        self,
        order_price: pl.Series | None=None,
        order_quantity: pl.Series | int | float=1,
        first_only: bool=False,
        long_only: bool=False,
        short_only: bool=False,
    ) -> pl.DataFrame:
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
        market_order_trade_condition = (
            (long_order & (prev_close <= opened_order_price)) |
            (short_order & (prev_close >= opened_order_price))
        )
        limit_order_trade_condition = (
            (long_order & (opened_order_price >= pl.col('low'))) |
            (short_order & (opened_order_price <= pl.col('high')))
        )
        trade_condition = market_order_trade_condition | limit_order_trade_condition
        # NOTE: the actual trade price is 'open', not prev_close
        df = df.with_columns(
            trade_price=(
                pl.when(market_order_trade_condition).then(pl.col('open'))
                .when(limit_order_trade_condition).then(opened_order_price)
                .otherwise(None)
            ),
            # if order size exceeds volume, trade size = volume * order side
            trade_size=(
                pl.when(trade_condition & (opened_order_size.abs() > pl.col('volume')))
                .then(pl.col('volume') * opened_order_side)
                # otherwise trade with order size
                .when(trade_condition)
                .then(opened_order_size)
                .otherwise(None)
            ),
        )

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

        return self.__class__(df)