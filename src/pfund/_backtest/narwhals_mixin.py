# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportReturnType=false, reportMissingTypeArgument=false, reportImplicitStringConcatenation=false, reportUnknownParameterType=false, reportArgumentType=false, reportCallIssue=false
from typing import Literal

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT, IntoSeries

from pfund._backtest.numba_kernel import backtest_loop_kernel


def _to_float64(series: nw.Series) -> np.ndarray:
    """Convert a narwhals Series to a numpy float64 array."""
    arr = series.to_numpy()
    return arr.astype(np.float64, copy=False)


def _series_to_positional_float64(
    input_series: IntoSeries, n: int, name: str
) -> np.ndarray:
    """Convert a native Series to a positional float64 array, with length validation.

    Uses .values (positional) instead of index-aligned operations to match
    the hybrid kernel's positional semantics.

    Args:
        input_series: A native Series (pandas or polars) to convert.
        n: Expected length of the series (must match dataframe length).
        name: Name of the series, used in error messages for length mismatches.

    Returns:
        A float64 numpy array of the series values.

    Raises:
        ValueError: If the series length does not match ``n``.
    """
    nw_series = nw.from_native(input_series, series_only=True)
    arr = _to_float64(nw_series)
    if len(arr) != n:
        raise ValueError(
            f"'{name}' length ({len(arr)}) does not match dataframe length ({n}). "
            f"Ensure '{name}' has the same number of rows as the dataframe."
        )
    return arr


class NarwhalsMixin:
    def create_signal(
        self: IntoDataFrameT,
        buy_condition: IntoSeries | None = None,
        sell_condition: IntoSeries | None = None,
        signal: IntoSeries | None = None,
        first_only: bool = False,
    ) -> IntoDataFrameT:
        """
        A signal is defined as a sequence of 1s and -1s, where 1 means a buy signal and -1 means a sell signal.
        All series arguments are native series (pandas or polars), matching the dataframe's backend.
        Args:
            buy_condition: condition to create a buy signal 1
            sell_condition: condition to create a sell signal -1
            signal: provides self-defined signals, buy_condition and sell_condition are ignored if provided
            first_only: only the first signal is remained in each signal sequence
                useful when only the first signal is treated as a true signal
        """
        df = nw.from_native(self)

        if signal is None:
            if buy_condition is None and sell_condition is None:
                raise ValueError("Either buy or sell condition must be provided")
            if buy_condition is not None and sell_condition is not None:
                buy = nw.from_native(buy_condition, series_only=True)
                sell = nw.from_native(sell_condition, series_only=True)
                overlaps = buy & sell
                if overlaps.any():
                    raise ValueError(
                        "Overlapping buy and sell condition detected.\n"
                        + "Please make sure that buy and sell conditions are mutually exclusive."
                    )
                df = df.with_columns(buy.alias("_buy"), sell.alias("_sell"))
                df = df.with_columns(
                    nw.when(nw.col("_buy"))
                    .then(1)
                    .when(nw.col("_sell"))
                    .then(-1)
                    .otherwise(None)
                    .alias("signal")
                ).drop("_buy", "_sell")
            else:
                cond = nw.from_native(
                    buy_condition if buy_condition is not None else sell_condition,
                    series_only=True,
                )
                value = 1 if buy_condition is not None else -1
                df = df.with_columns(cond.alias("_cond"))
                df = df.with_columns(
                    nw.when(nw.col("_cond")).then(value).otherwise(None).alias("signal")
                ).drop("_cond")
        else:
            signal_series = nw.from_native(signal, series_only=True)
            assert signal_series.drop_nulls().unique().is_in([1, -1]).all(), (
                "'signal' must only contain 1, -1, null"
            )
            df = df.with_columns(signal_series.alias("signal"))

        # _signal_change: True where the forward-filled signal differs from its previous value
        signal_ffill = nw.col("signal").fill_null(strategy="forward")
        df = df.with_columns(
            (
                (signal_ffill != signal_ffill.shift(1)).fill_null(True)
                & ~signal_ffill.is_null()
            ).alias("_signal_change")
        )

        if first_only:
            df = df.with_columns(
                nw.when(nw.col("_signal_change"))
                .then(nw.col("signal"))
                .otherwise(None)
                .alias("signal")
            )

        return self.__class__(df.to_native())

    def open_position(
        self: IntoDataFrameT,
        order_price: IntoSeries | None = None,
        order_quantity: IntoSeries | int | float = 1,
        first_only: bool = False,
        long_only: bool = False,
        short_only: bool = False,
        fill_price: Literal["open", "close"] = "close",
    ) -> IntoDataFrameT:
        """
        Sets up position-opening orders; the actual backtest runs in backtest_loop().
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
            long_only: orders in signal=-1 only close the existing long position, the position will not be flipped
                useful for long-only strategy with signal=-1 to close the position
            short_only: orders in signal=1 only close the existing short position, the position will not be flipped
                useful for short-only strategy with signal=1 to close the position
            fill_price: fill price for market orders.
                An order is placed at bar N's close. 'close' fills at bar N's close price,
                'open' fills at bar N+1's open price.
                Applies to regular market orders (aggressive vs close).
                Gap-through limits always fill at bar N+1 open.
                In-range limit orders fill at their limit price.
                Default is 'close', which avoids gap exposure (e.g. overnight gaps on daily bars).
        """
        assert "signal" in self.columns, (
            "No 'signal' column is found, please use create_signal() first"
        )
        assert not (long_only and short_only), (
            "Cannot be long_only and short_only at the same time"
        )
        assert fill_price in ("open", "close"), "'fill_price' must be 'open' or 'close'"

        # everything is delayed to backtest_loop(), only store the inputs here
        self._open_position_inputs = {
            "order_price": order_price,
            "order_quantity": order_quantity,
            "first_only": first_only,
            "long_only": long_only,
            "short_only": short_only,
            "fill_price": fill_price,
        }
        # Keep stored inputs on the same object; returning a new
        # BacktestDataFrame would reset these attributes in __init__.
        return self

    def close_position(
        self: IntoDataFrameT,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        time_window: int | None = None,
        fill_price: Literal["open", "close"] = "close",
    ) -> IntoDataFrameT:
        """
        Sets up position-closing conditions; the actual backtest runs in backtest_loop().
        Conceptually, this function places stop market orders at the end of each bar, after placing orders in open_position().
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
        """
        if take_profit is not None:
            assert take_profit > 0, "'take_profit' must be positive"
        if stop_loss is not None:
            stop_loss = abs(stop_loss)
            assert 1 > stop_loss > 0, "'stop_loss' must be between 0 and 1"
        if time_window is not None:
            assert isinstance(time_window, int) and time_window > 0, (
                "'time_window' must be a positive integer"
            )
        assert fill_price in ("open", "close"), "'fill_price' must be 'open' or 'close'"

        # everything is delayed to backtest_loop(), only store the inputs here
        self._close_position_inputs = {
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "time_window": time_window,
            "fill_price": fill_price,
        }
        # Keep stored inputs on the same object; returning a new
        # BacktestDataFrame would reset these attributes in __init__.
        return self

    def backtest_loop(
        self: IntoDataFrameT,
        trailing_stop: float | None = None,
    ) -> IntoDataFrameT:
        """Runs the bar-by-bar backtest loop via the numba kernel.

        This is the final step in the backtest chain:
            df.create_signal(...).open_position(...).close_position(...).backtest_loop(...)

        Args:
            trailing_stop: Trailing stop percentage (0 < value < 1).
                A dynamic stop that follows the price in a favorable direction but does not move against it.
                When price reverses by the trailing percentage from its best level, a market order is triggered to close the position.
                When set, replaces the fixed stop_loss from close_position().
        """
        # ================================================================
        # Validate preconditions
        # ================================================================
        if not self._open_position_inputs:
            raise ValueError(
                "backtest_loop() requires open_position() to be called first. "
                "Chain: df.create_signal(...).open_position(...).close_position(...).backtest_loop()"
            )
        if not self._close_position_inputs:
            raise ValueError(
                "backtest_loop() requires close_position() to be called first. "
                "Chain: df.create_signal(...).open_position(...).close_position(...).backtest_loop()"
            )

        df = nw.from_native(self)
        native_backend = nw.get_native_namespace(df)
        n = len(df)

        # ================================================================
        # Gather stored inputs from open_position() and close_position()
        # ================================================================
        order_price_input = self._open_position_inputs["order_price"]
        order_quantity_input = self._open_position_inputs["order_quantity"]
        first_only = self._open_position_inputs["first_only"]
        long_only = self._open_position_inputs["long_only"]
        short_only = self._open_position_inputs["short_only"]
        open_fill_price = self._open_position_inputs["fill_price"]
        take_profit = self._close_position_inputs["take_profit"]
        stop_loss = self._close_position_inputs["stop_loss"]
        time_window = self._close_position_inputs["time_window"]
        close_fill_price = self._close_position_inputs["fill_price"]

        # ================================================================
        # Convert all inputs to numpy float64 arrays for numba
        # ================================================================
        open_arr = _to_float64(df.get_column("open"))
        high_arr = _to_float64(df.get_column("high"))
        low_arr = _to_float64(df.get_column("low"))
        close_arr = _to_float64(df.get_column("close"))
        volume_arr = _to_float64(df.get_column("volume"))
        signal_arr = _to_float64(df.get_column("signal"))

        # order_price: None → use close prices; Series → convert positionally with validation
        if order_price_input is None:
            order_price_arr = close_arr.copy()
        else:
            order_price_arr = _series_to_positional_float64(
                order_price_input, n, "order_price"
            )
            # Validate: order_price must be positive or nan
            valid = np.isnan(order_price_arr) | (order_price_arr > 0)
            assert valid.all(), "'order_price' must be positive or nan"

        # order_quantity: scalar → broadcast to array; Series → convert positionally with validation
        if isinstance(order_quantity_input, (int, float, np.number)):
            assert order_quantity_input > 0, "'order_quantity' must be positive"
            order_quantity_arr = np.full(n, float(order_quantity_input))
        else:
            order_quantity_arr = _series_to_positional_float64(
                order_quantity_input, n, "order_quantity"
            )
            # Validate: order_quantity must be positive or nan
            valid = np.isnan(order_quantity_arr) | (order_quantity_arr > 0)
            assert valid.all(), "'order_quantity' values must be positive or nan"

        # Scalars: None → nan for numba (tp/sl are floats, nan is the natural sentinel).
        # time_window is an integer bar count, so use -1 as sentinel instead of np.nan;
        # the numba kernel uses `time_window > 0` to decide if the feature is enabled.
        tp = np.nan if take_profit is None else float(take_profit)
        sl = np.nan if stop_loss is None else float(stop_loss)
        tw = -1 if time_window is None else int(time_window)
        if trailing_stop is not None:
            assert 0 < trailing_stop < 1, (
                "'trailing_stop' must be between 0 and 1 (exclusive)"
            )
        ts = np.nan if trailing_stop is None else float(trailing_stop)
        # fill_price: convert 'open'/'close' to bool for numba (True = fill at open)
        market_fill_at_open = open_fill_price == "open"
        exit_market_fill_at_open = close_fill_price == "open"

        # ================================================================
        # Run numba kernel
        # ================================================================
        (
            order_price_out,
            order_size_out,
            trade_price_out,
            trade_size_out,
            position_out,
            avg_price_out,
            stop_price_out,
        ) = backtest_loop_kernel(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            volume_arr,
            signal_arr,
            order_price_arr,
            order_quantity_arr,
            first_only,
            long_only,
            short_only,
            tp,
            sl,
            tw,
            ts,
            market_fill_at_open,
            exit_market_fill_at_open,
            n,
        )

        # ================================================================
        # Assign output arrays back to dataframe columns
        # ================================================================
        def _make_series(name: str, values: np.ndarray) -> nw.Series:
            return nw.new_series(name, values, nw.Float64, backend=native_backend)

        result_df = df.with_columns(
            _make_series("order_price", order_price_out),
            _make_series("order_size", order_size_out),
            _make_series("trade_price", trade_price_out),
            _make_series("trade_size", trade_size_out),
            _make_series("position", position_out),
            _make_series("avg_price", avg_price_out),
        )

        # Only add stop_price column if any stops actually triggered
        if not np.all(np.isnan(stop_price_out)):
            result_df = result_df.with_columns(
                _make_series("stop_price", stop_price_out),
            )

        # Wrap back into the custom BacktestDataframe class (to_native() returns plain pd/pl DataFrame)
        return self.__class__(result_df.to_native())
