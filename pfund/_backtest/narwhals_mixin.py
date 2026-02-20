# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportReturnType=false, reportMissingTypeArgument=false, reportImplicitStringConcatenation=false, reportUnknownParameterType=false, reportArgumentType=false, reportCallIssue=false
import numpy as np
import narwhals as nw
from narwhals.typing import IntoDataFrameT, IntoSeries

from pfund.enums import BacktestMode
from pfund._backtest.numba_kernel import backtest_loop_kernel


def _to_float64(series: nw.Series) -> np.ndarray:
    """Convert a narwhals Series to a numpy float64 array."""
    arr = series.to_numpy()
    return arr.astype(np.float64, copy=False)


def _series_to_positional_float64(input_series: IntoSeries, n: int, name: str) -> np.ndarray:
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
    def backtest_loop(
        self: IntoDataFrameT,
        trailing_stop: float | None = None,
    ) -> IntoDataFrameT:
        '''Runs the bar-by-bar hybrid backtest loop via the numba kernel.

        This is the final step in the hybrid backtest chain:
            df.create_signal(...).open_position(...).close_position(...).backtest_loop(...)

        Args:
            trailing_stop: Trailing stop percentage (0 < value < 1).
                A dynamic stop that follows the price in a favorable direction but does not move against it.
                When price reverses by the trailing percentage from its best level, a market order is triggered to close the position.
                When set, replaces the fixed stop_loss from close_position().
        '''
        if self._backtest_mode != BacktestMode.HYBRID:
            raise ValueError("backtest_loop() is only available in backtest mode 'hybrid'")


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
        order_price_input = self._open_position_inputs['order_price']
        order_quantity_input = self._open_position_inputs['order_quantity']
        first_only = self._open_position_inputs['first_only']
        long_only = self._open_position_inputs['long_only']
        short_only = self._open_position_inputs['short_only']
        open_fill_price = self._open_position_inputs['fill_price']
        take_profit = self._close_position_inputs['take_profit']
        stop_loss = self._close_position_inputs['stop_loss']
        time_window = self._close_position_inputs['time_window']
        close_fill_price = self._close_position_inputs['fill_price']


        # ================================================================
        # Convert all inputs to numpy float64 arrays for numba
        # ================================================================
        open_arr = _to_float64(df.get_column('open'))
        high_arr = _to_float64(df.get_column('high'))
        low_arr = _to_float64(df.get_column('low'))
        close_arr = _to_float64(df.get_column('close'))
        volume_arr = _to_float64(df.get_column('volume'))
        signal_arr = _to_float64(df.get_column('signal'))

        # order_price: None → use close prices; Series → convert positionally with validation
        if order_price_input is None:
            order_price_arr = close_arr.copy()
        else:
            order_price_arr = _series_to_positional_float64(order_price_input, n, 'order_price')
            # Validate: order_price must be positive or nan
            valid = np.isnan(order_price_arr) | (order_price_arr > 0)
            assert valid.all(), "'order_price' must be positive or nan"

        # order_quantity: scalar → broadcast to array; Series → convert positionally with validation
        if isinstance(order_quantity_input, (int, float, np.number)):
            assert order_quantity_input > 0, "'order_quantity' must be positive"
            order_quantity_arr = np.full(n, float(order_quantity_input))
        else:
            order_quantity_arr = _series_to_positional_float64(order_quantity_input, n, 'order_quantity')
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
            assert 0 < trailing_stop < 1, "'trailing_stop' must be between 0 and 1 (exclusive)"
        ts = np.nan if trailing_stop is None else float(trailing_stop)
        # fill_price: convert 'open'/'close' to bool for numba (True = fill at open)
        market_fill_at_open = (open_fill_price == 'open')
        exit_market_fill_at_open = (close_fill_price == 'open')


        # ================================================================
        # Run numba kernel
        # ================================================================
        (
            order_price_out, order_size_out,
            trade_price_out, trade_size_out,
            position_out, avg_price_out, stop_price_out,
        ) = backtest_loop_kernel(
            open_arr, high_arr, low_arr, close_arr, volume_arr,
            signal_arr,
            order_price_arr, order_quantity_arr,
            first_only, long_only, short_only,
            tp, sl, tw,
            ts,
            market_fill_at_open, exit_market_fill_at_open,
            n,
        )


        # ================================================================
        # Assign output arrays back to dataframe columns
        # ================================================================
        def _make_series(name: str, values: np.ndarray) -> nw.Series:
            return nw.new_series(name, values, nw.Float64, backend=native_backend)

        result_df = df.with_columns(
            _make_series('order_price', order_price_out),
            _make_series('order_size', order_size_out),
            _make_series('trade_price', trade_price_out),
            _make_series('trade_size', trade_size_out),
            _make_series('position', position_out),
            _make_series('avg_price', avg_price_out),
        )

        # Only add stop_price column if any stops actually triggered
        if not np.all(np.isnan(stop_price_out)):
            result_df = result_df.with_columns(
                _make_series('stop_price', stop_price_out),
            )

        # Wrap back into the custom BacktestDataframe class (to_native() returns plain pd/pl DataFrame)
        return self.__class__(result_df.to_native(), backtest_mode=self._backtest_mode)
