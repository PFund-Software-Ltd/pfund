# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportReturnType=false, reportMissingTypeArgument=false, reportImplicitStringConcatenation=false, reportUnknownParameterType=false, reportArgumentType=false, reportCallIssue=false
import datetime
from collections.abc import Callable
from typing import Literal

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT, IntoSeries

from pfund.datas.stores.market_data_store import MarketDataStore
from pfund._backtest.product_backtest_kernel import backtest_loop_kernel

# what a registration is keyed by, minus its data_range: a (resolution, product)
# combo for product backtesting, a bare product string for portfolio
_Identity = tuple[str, str] | str

# (resolution, product) = a combo; one entry per create_signal() call
# (= one time SEGMENT of a combo). Registration controls INSTRUCTIONS, never
# data: backtest() always runs the kernel over a combo's FULL rows and
# scatters each segment's values onto them, so positions carry across
# CONTIGUOUS segments and stops are evaluated on every covered bar; a position
# on an uncovered (gap) row is closed — out of the universe (close-on-exit).
_registry: dict[tuple, dict] = {}
# ((start_date, end_date), "1d", "BTC"): {  # key = (data_range, *combo); combo in PIVOT_COLS order
#   "signal": np.array([1.0, nan, -1.0, ...]),  # positional over the entry's rows
#   "dates":  np.array([date1, date2, ...]),  # the entry's rows' dates, pin row alignment
#   "open":   {"order_quantity": 2, "order_price": None, "first_only": False, ...},  # open_position kwargs
#   "close":  {"stop_loss": 0.05, "take_profit": None, ...},  # close_position kwargs
# }
# open_position()/close_position() attach to the combo's most recently
# registered segment:
_latest_key_by_combo: dict[tuple, tuple] = {}

_PATTERN = (
    "# fixed period — configure each (resolution, product) combo once:\n"
    "for resolution, product in df.select('resolution', 'product').unique().rows():\n"
    "    (\n"
    "        df\n"
    "        .filter(pl.col('resolution') == resolution, pl.col('product') == product)\n"
    "        .create_signal(...)\n"
    "        .open_position(...)\n"
    "        .close_position(...)\n"
    "    )\n"
    "\n"
    "# dynamic periods (optional) — point-in-time configuration per period:\n"
    "stride = '1d'  # trading period: '1d', '1w', '1mo', ...\n"
    "periods = sorted(df['date'].dt.truncate(stride).unique().to_list())\n"
    "for period in periods:\n"
    "    # expanding point-in-time window: history up to the current period\n"
    "    periodic_df = df.filter(pl.col('date').dt.truncate(stride) <= period)\n"
    "    universe = sorted(periodic_df['product'].unique().to_list())\n"
    "    # NOTE: add custom screening logic to shortlist the universe\n"
    "    for product in universe:\n"
    "        (\n"
    "            periodic_df\n"
    "            .filter(pl.col('product') == product)\n"
    "            .create_signal(..., data_range=(period_start, period_end))\n"
    "            .open_position(...)\n"
    "            .close_position(...)\n"
    "        )\n"
    "\n"
    "df.backtest()"
)


def _clear_registry() -> None:
    _registry.clear()
    _latest_key_by_combo.clear()


def _get_current_combo(df: nw.DataFrame) -> tuple:
    """Return the single (resolution, product) combo's values in PIVOT_COLS order."""
    combos = df.select(*MarketDataStore.PIVOT_COLS).unique().rows()
    if len(combos) != 1:
        raise ValueError(
            "Multiple (resolution, product)s detected — backtest one (resolution, product) at a time:\n"
            + _PATTERN
        )
    return combos[0]


def _get_registry_key(
    df: nw.DataFrame, data_range: tuple[datetime.date, datetime.date] | None
) -> tuple[tuple[datetime.date, datetime.date], str, str]:
    """Build the real registry key (data_range, resolution, product) for one
    create_signal() call — the combo from the df (which must hold exactly one
    resolution and one product), the range as passed (None → the df's
    first/last dates)."""
    resolutions = df.get_column("resolution").unique().to_list()
    products = df.get_column("product").unique().to_list()
    if len(resolutions) != 1 or len(products) != 1:
        raise ValueError(
            "Multiple (resolution, product)s detected — backtest one (resolution, product) at a time:\n"
            + _PATTERN
        )
    if data_range is None:
        date_col = df.get_column("date")
        data_range = (date_col.item(0), date_col.item(len(date_col) - 1))
    return (data_range, resolutions[0], products[0])


def _validate_data_range(
    data_range: tuple[datetime.date, datetime.date],
) -> tuple[datetime.date, datetime.date]:
    if not (isinstance(data_range, tuple) and len(data_range) == 2):
        raise ValueError("'data_range' must be a (start_date, end_date) tuple")
    start, end = data_range
    for bound in (start, end):
        # datetime.datetime is a datetime.date subclass: both are accepted
        if not isinstance(bound, datetime.date):
            raise ValueError(
                f"'data_range' bounds must be date/datetime, got {type(bound).__name__}"
            )
    if np.datetime64(start) > np.datetime64(end):
        raise ValueError(f"'data_range' start {start} is after its end {end}")
    return start, end


def _data_range_mask(
    date_arr: np.ndarray, start: datetime.date, end: datetime.date
) -> np.ndarray:
    """Boolean mask of rows within [start, end], both bounds inclusive.

    A plain-date bound compares at DAY precision, covering the whole day
    regardless of the bars' intraday timestamps; a datetime bound compares
    exactly.
    """

    def _compare(bound: datetime.date, is_start: bool) -> np.ndarray:
        if isinstance(bound, datetime.datetime):
            arr, b = date_arr, np.datetime64(bound)
        else:
            arr, b = date_arr.astype("datetime64[D]"), np.datetime64(bound, "D")
        return (arr >= b) if is_start else (arr <= b)

    return _compare(start, True) & _compare(end, False)


def _to_float64(series: nw.Series) -> np.ndarray:
    """Convert a narwhals Series to a numpy float64 array."""
    arr = series.to_numpy()
    return arr.astype(np.float64, copy=False)


def _slice_series_to_data_range(
    df: nw.DataFrame,
    data_range: tuple[datetime.date, datetime.date],
    series_input: IntoSeries | int | float | None,
    name: str,
) -> IntoSeries | int | float | None:
    """Slice a full-window series down to its data_range rows.

    Mirrors create_signal: every series passed in the chain is positional over
    the SAME df the method was called on (the expanding point-in-time window),
    so order_price/order_quantity are built the same shape as the conditions —
    one rule for the whole chain. The full length is validated, then only the
    data_range rows (the rows actually registered) are kept, so the stored
    series lines up positionally with the segment. Scalars and None pass through
    unchanged.
    """
    if series_input is None or isinstance(series_input, (int, float, np.number)):
        return series_input
    nw_series = nw.from_native(series_input, series_only=True)
    n = len(df)
    if len(nw_series) != n:
        raise ValueError(
            f"'{name}' length ({len(nw_series)}) must match the dataframe length "
            f"({n}) — a series passed to open_position() is positional over the "
            "same df create_signal() was called on (it is sliced internally to "
            "the data_range rows)."
        )
    start, end = _validate_data_range(data_range)
    mask = _data_range_mask(df.get_column("date").to_numpy(), start, end)
    backend = nw.get_native_namespace(nw_series)
    nmask = nw.new_series("_mask", mask, nw.Boolean, backend=backend)
    return nw_series.filter(nmask).to_native()


def _group_positions(pivot_arrs: list[np.ndarray], n: int) -> dict[tuple, np.ndarray]:
    """Map each (resolution, product) combo to its row positions in ONE pass.

    A stable lexsort over the pivot columns groups identical combos into
    contiguous runs (O(n log n)); within a run the original row order is
    preserved, so the returned positions are in ascending row order — matching
    the old boolean-mask semantics that backtest()'s date check relies on.
    Replaces the previous O(combos * n) per-combo full-df masking.
    """
    if n == 0:
        return {}
    # lexsort's LAST key is primary; reverse so PIVOT_COLS[0] sorts first.
    order = np.lexsort(tuple(pivot_arrs[::-1]))
    sorted_cols = [a[order] for a in pivot_arrs]
    change = np.zeros(n, dtype=np.bool_)
    change[0] = True
    for sc in sorted_cols:
        change[1:] |= sc[1:] != sc[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)
    positions: dict[tuple, np.ndarray] = {}
    for s, e in zip(starts, ends):
        # normalize numpy scalars to python types so the key matches the
        # registry key built from df.select(*PIVOT_COLS).unique().rows();
        # object-dtype arrays (pandas strings) already yield python objects
        key = tuple(
            v.item() if isinstance(v, np.generic) else v
            for v in (col[s] for col in sorted_cols)
        )
        positions[key] = order[s:e]
    return positions


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


def _scatter_order_inputs(
    open_inputs: dict,
    seg_rows: np.ndarray,
    combo_close: np.ndarray,
    order_price_arr: np.ndarray,
    order_quantity_arr: np.ndarray,
) -> None:
    """Scatter one segment's order_price/order_quantity onto its rows.

    seg_rows = the segment's row indices within the combo's full row sequence.
    Series args are positional over those rows; scalars are broadcast. Rows
    covered by no segment stay nan — the kernel skips order placement on nan
    price/quantity (no instruction).
    """
    num_rows = len(seg_rows)
    order_price_input = open_inputs["order_price"]
    # order_price: None → use close prices (market orders); Series → convert
    # positionally with validation
    if order_price_input is None:
        order_price_arr[seg_rows] = combo_close[seg_rows]
    else:
        arr = _series_to_positional_float64(order_price_input, num_rows, "order_price")
        # Validate: order_price must be positive or nan
        valid = np.isnan(arr) | (arr > 0)
        if not valid.all():
            raise ValueError("'order_price' must be positive or nan")
        order_price_arr[seg_rows] = arr

    order_quantity_input = open_inputs["order_quantity"]
    # order_quantity: scalar → broadcast; Series → convert positionally with validation
    if isinstance(order_quantity_input, (int, float, np.number)):
        if not order_quantity_input > 0:
            raise ValueError("'order_quantity' must be positive")
        order_quantity_arr[seg_rows] = float(order_quantity_input)
    else:
        arr = _series_to_positional_float64(
            order_quantity_input, num_rows, "order_quantity"
        )
        # Validate: order_quantity must be positive or nan
        valid = np.isnan(arr) | (arr > 0)
        if not valid.all():
            raise ValueError("'order_quantity' values must be positive or nan")
        order_quantity_arr[seg_rows] = arr


def _resolve_targets(
    registry: dict[tuple, dict],
    identity: _Identity,
    key_identity: Callable[[tuple], _Identity],
    df: nw.DataFrame,
    data_range: tuple[datetime.date, datetime.date],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve one registration call's target rows and enforce register-once rules.

    Shared by product (create_signal) and portfolio (create_weight): both
    register one identity's rows for a (already-resolved) data_range into a
    module-global registry keyed by (data_range, *identity). The caller's
    _get_registry_key resolves None→the df's first/last span, so the range
    here is always concrete.

    First the df is validated: it must carry a 'date' column (it pins the rows
    so backtest() can verify alignment) and be sorted ascending by it —
    registration is positional (signal ffill/shift; a scalar weight lands on the
    range's last row) and a segment's first row anchors backtest()'s
    extend-forward rule, so an unsorted df would silently misalign. Then the
    in-range rows are covered and any OVERLAP with an already-registered range
    of this identity is rejected (a row's instruction/weight is registered once
    — the past is never rewritten).

    Duplicate dates in the covered rows are rejected: they would double-scatter
    onto the same row at backtest() (searchsorted maps every duplicate to its
    first occurrence — last value silently wins).

    Args:
        registry: the caller's module-global registry dict.
        identity: the combo/product this call registers; its str() prefixes the
            error messages and it is matched against existing keys via key_identity.
        key_identity: maps a stored registry key to its identity, so entries of the
            SAME identity can be found (product: ``k[1:]``; portfolio: ``k[1]``).
        df: the df the registration method was called on (must hold a sorted
            'date' column).
        data_range: the resolved (start, end), both inclusive.

    Returns:
        (target_pos, target_dates) — the covered rows' positions in the df and
        their dates (to pin alignment).
    """
    if "date" not in df.columns:
        raise ValueError(
            "backtesting requires a 'date' column — it pins the registered rows "
            "so backtest() can verify row alignment"
        )
    date_col = df.get_column("date")
    if not date_col.is_sorted():
        raise ValueError(
            "the dataframe passed to backtesting must be sorted by 'date' (ascending)"
        )
    start, end = _validate_data_range(data_range)
    date_arr = date_col.to_numpy()
    target_pos = np.flatnonzero(_data_range_mask(date_arr, start, end))
    target_dates = date_arr[target_pos]
    if len(np.unique(target_dates)) != len(target_dates):
        raise ValueError(
            f"{identity}: duplicate dates in the registered rows — backtesting "
            "requires one row per date per product (a single resolution)"
        )

    # a row is registered once — the past is never rewritten
    for registered_key, entry in registry.items():
        if key_identity(registered_key) != identity:
            continue
        if len(target_dates) and np.isin(target_dates, entry["dates"]).any():
            raise ValueError(
                f"{identity}: data_range {data_range} overlaps the already-registered "
                f"{registered_key[0]} — a row can only be registered once"
            )
    return target_pos, target_dates


class ProductBacktestMixin:
    def create_signal(
        self: IntoDataFrameT,
        buy_condition: IntoSeries | None = None,
        sell_condition: IntoSeries | None = None,
        signal: IntoSeries | None = None,
        first_only: bool = False,
        data_range: tuple[datetime.date, datetime.date] | None = None,
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
            data_range: (start_date, end_date), both inclusive — the rows this
                call covers (one time SEGMENT of the combo). Conditions/signals
                (incl. first_only thinning) are still computed over the WHOLE
                df (an expanding point-in-time window provides history for
                indicators); only the rows inside data_range are registered.
                Registration controls instructions, never data: backtest()
                always runs the kernel over the combo's FULL rows, so positions
                carry across CONTIGUOUS segments and stops are evaluated on
                every covered bar. A row covered by NO segment is out of the
                universe: an open position there is CLOSED at that bar's close
                (close-on-exit) — to hold across a gap, register those rows
                (a nan signal = hold while in the universe).
                A range overlapping an already-registered range of the same
                combo raises — a row's instruction is registered once. A range
                selecting no rows (e.g. a product not yet listed in that
                period) is a no-op.
                None (default) → configure over ALL rows of this df (full
                span). Registration is still register-once: this raises if it
                would overlap an already-registered segment of the combo.

        Pure REGISTRATION step: computes this combo's signal series and
        registers the rows in data_range (with their dates) in the session
        registry — no column is written to any df and self is returned
        unchanged. The signal appears as a column in the result df produced by
        backtest(). Must be called on a df holding exactly one
        (product, resolution) combo (a bare OHLCV df is one combo).
        """
        df = nw.from_native(self)
        key = _get_registry_key(df, data_range)
        data_range, resolution, product = key
        combo = (resolution, product)
        target_pos, target_dates = _resolve_targets(
            _registry,
            combo,
            lambda k: k[1:],
            df,
            data_range,
        )

        # compute signal series
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
            if not signal_series.drop_nulls().unique().is_in([1, -1]).all():
                raise ValueError("'signal' must only contain 1, -1, null")
            df = df.with_columns(signal_series.alias("signal"))

        # _signal_change: True where the forward-filled signal differs from its
        # previous value. The df is a single combo, so no per-combo scoping is needed.
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
        signal_arr = _to_float64(df.get_column("signal"))

        # register signal series and its target dates
        _registry[key] = {
            "dates": target_dates,
            "signal": signal_arr[target_pos],
            "open": None,
            "close": None,
        }
        _latest_key_by_combo[combo] = key

        # dynamically add backtest methods to the dataframe class
        df_class = type(self)
        df_class.open_position = ProductBacktestMixin.open_position
        df_class.close_position = ProductBacktestMixin.close_position
        df_class.backtest = ProductBacktestMixin.backtest
        return self

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
        Sets up position-opening orders; the actual backtest runs in backtest().
        Conceptually, this function places orders at the end of bar/candlestick N.
        For example, for a buy order:
        - If the order price >= close price of bar N, it is a market order,
            by assuming that the close price is the current best price; otherwise it is a limit order.
        Then the orders are opened at the beginning of bar N+1,
        and filled during bar N+1 if high >= order price >= low.
        If bar N+1 gaps through the limit (order price is outside [low, high]),
        the order is treated as a marketable gap-through and fills at N+1 open.
        Opened orders are considered as cancelled at the end of bar N+1 if not filled.
        Pure REGISTRATION step: validates and registers the kwargs for this
        combo's MOST RECENTLY registered segment (create_signal must have been
        called on it first); self is returned unchanged. All params may vary
        per segment. Series args (order_price/order_quantity) are positional
        over the SAME df create_signal() was called on (the full expanding
        point-in-time window) and are sliced internally to the segment's
        data_range rows — build them the same shape as create_signal's
        conditions; no manual slicing needed.
        Args:
            order_price: price to place the order.
                If None, use 'close' price (market order).
                A Series may contain nan = no order on that bar; this also
                gates long_only/short_only closes — a nan on a close bar
                means hold through the signal (stops stay live).
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
        if long_only and short_only:
            raise ValueError("Cannot be long_only and short_only at the same time")
        if fill_price not in ("open", "close"):
            raise ValueError("'fill_price' must be 'open' or 'close'")
        # bool is an int subclass: True would silently become quantity 1.0
        if isinstance(order_quantity, bool):
            raise ValueError(
                "'order_quantity' must be a number or a Series, not a bool"
            )

        df = nw.from_native(self)
        combo = _get_current_combo(df)
        latest_key = _latest_key_by_combo.get(combo)
        if latest_key is None:
            raise ValueError(
                "open_position() requires create_signal() to be called first:\n"
                + _PATTERN
            )
        # series are positional over the SAME df create_signal() was called on
        # (the expanding window); slice them down to the segment's data_range
        # rows here, exactly as create_signal does for its signal series
        data_range = latest_key[0]
        order_price = _slice_series_to_data_range(
            df, data_range, order_price, "order_price"
        )
        order_quantity = _slice_series_to_data_range(
            df, data_range, order_quantity, "order_quantity"
        )
        _registry[latest_key]["open"] = {
            "order_price": order_price,
            "order_quantity": order_quantity,
            "first_only": first_only,
            "long_only": long_only,
            "short_only": short_only,
            "fill_price": fill_price,
        }
        return self

    def close_position(
        self: IntoDataFrameT,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        trailing_stop: float | None = None,
        time_window: int | None = None,
        fill_price: Literal["open", "close"] = "close",
    ) -> IntoDataFrameT:
        """
        Sets up position-closing conditions; the actual backtest runs in backtest().
        Conceptually, this function places stop market orders at the end of each bar, after placing orders in open_position().
        Pure REGISTRATION step: validates and registers the kwargs for this
        combo's MOST RECENTLY registered segment (create_signal must have been
        called on it first); self is returned unchanged. All params may vary
        per segment; each segment's params govern from its first row until the
        next segment begins. A position is closed on the first row covered by
        no segment (close-on-exit), so stops only ever apply within covered
        rows.
        Args:
            take_profit: take profit percentage (e.g. 0.1 = 10%).
            stop_loss: stop loss percentage between 0 and 1 (e.g. 0.05 = 5%).
            trailing_stop: trailing stop percentage between 0 and 1 (e.g. 0.05 = 5%).
                A dynamic stop that follows the price in a favorable direction but does not move against it.
                When price reverses by the trailing percentage from its best level since entry,
                a market order is triggered to close the position.
                Mutually exclusive with stop_loss.
            time_window: max number of bars to hold a position before auto-closing.
            fill_price: fill price for market close orders (immediately triggered SL/TP and time_window).
                These trigger at bar N's close. 'close' fills at bar N's close price,
                'open' fills at bar N+1's open price.
                Gap-through immediate SL/TP (stop outside [low, high]) always fill at bar N+1 open.
                Non-immediately triggered SL/TP (high/low breach during bar) always fill at their trigger price.
                Default is 'close', which avoids gap exposure (e.g. overnight gaps on daily bars).
        """
        if take_profit is not None and not take_profit > 0:
            raise ValueError("'take_profit' must be positive")
        if stop_loss is not None:
            stop_loss = abs(stop_loss)
            if not 1 > stop_loss > 0:
                raise ValueError("'stop_loss' must be between 0 and 1")
        if trailing_stop is not None:
            if not 0 < trailing_stop < 1:
                raise ValueError("'trailing_stop' must be between 0 and 1 (exclusive)")
            if stop_loss is not None:
                raise ValueError(
                    "'stop_loss' and 'trailing_stop' are mutually exclusive"
                )
        if time_window is not None and not (
            isinstance(time_window, int) and time_window > 0
        ):
            raise ValueError("'time_window' must be a positive integer")
        if fill_price not in ("open", "close"):
            raise ValueError("'fill_price' must be 'open' or 'close'")

        combo = _get_current_combo(nw.from_native(self))
        latest_key = _latest_key_by_combo.get(combo)
        if latest_key is None:
            raise ValueError(
                "close_position() requires create_signal() to be called first:\n"
                + _PATTERN
            )
        _registry[latest_key]["close"] = {
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "trailing_stop": trailing_stop,
            "time_window": time_window,
            "fill_price": fill_price,
        }
        return self

    def backtest(self: IntoDataFrameT) -> IntoDataFrameT:
        if not _registry:
            raise ValueError(
                "backtest() cannot run, setup should have been called properly:\n"
                + _PATTERN
            )

        df = nw.from_native(self)
        native_backend = nw.get_native_namespace(df)
        n = len(df)

        open_arr = _to_float64(df.get_column("open"))
        high_arr = _to_float64(df.get_column("high"))
        low_arr = _to_float64(df.get_column("low"))
        close_arr = _to_float64(df.get_column("close"))
        volume_arr = _to_float64(df.get_column("volume"))
        date_arr = df.get_column("date").to_numpy()

        # Group rows by combo once (O(n log n))
        positions_by_combo = _group_positions(
            [df.get_column(c).to_numpy() for c in MarketDataStore.PIVOT_COLS], n
        )

        # Outputs: nan = row not backtested (its combo was never configured)
        signal_out = np.full(n, np.nan)
        order_price_out = np.full(n, np.nan)
        order_size_out = np.full(n, np.nan)
        trade_price_out = np.full(n, np.nan)
        trade_size_out = np.full(n, np.nan)
        position_out = np.full(n, np.nan)
        avg_price_out = np.full(n, np.nan)
        stop_price_out = np.full(n, np.nan)
        output_arrs = (
            order_price_out,
            order_size_out,
            trade_price_out,
            trade_size_out,
            position_out,
            avg_price_out,
            stop_price_out,
        )

        # ================================================================
        # Run the numba kernel ONCE per combo (resolution, product) over the
        # combo's FULL rows; each registered segment scatters its values onto
        # per-bar input arrays (registration controls instructions, not data:
        # positions carry across segments, stops are evaluated on every bar)
        #
        # Regroup the flat registry into {combo -> [(date_range, reg), ...]},
        # bucketing every segment under its (resolution, product):
        #   _registry = {
        #       (jan, "1d", "AAPL"): reg1,  reg = {"signal": +1, "stop_loss": 0.02, ...},
        #       (feb, "1d", "AAPL"): reg2,
        #       (jan, "1d", "TSLA"): reg3,
        #   }
        #   entries_by_combo = {
        #       ("1d", "AAPL"): [(jan, reg1), (feb, reg2)],   # 2 segments
        #       ("1d", "TSLA"): [(jan, reg3)],                # 1 segment
        #   }
        # ================================================================
        entries_by_combo: dict[tuple, list[tuple]] = {}
        for registry_key, reg in _registry.items():
            entries_by_combo.setdefault(registry_key[1:], []).append(
                (registry_key[0], reg)
            )

        for combo, entries in entries_by_combo.items():
            # positions of this combo's rows (ascending), computed once above
            pos: np.ndarray | None = positions_by_combo.get(combo)
            if pos is None:
                raise ValueError(
                    f"{combo} has no rows in the dataframe passed to backtest()"
                )
            combo_dates = date_arr[pos]
            combo_close = close_arr[pos]
            num_rows = len(pos)
            if num_rows > 1 and (combo_dates[1:] == combo_dates[:-1]).any():
                raise ValueError(
                    f"{combo}: duplicate dates in the dataframe passed to backtest() — "
                    "product backtesting requires one row per date per "
                    "(resolution, product)"
                )

            # ------------------------------------------------------------
            # Resolve each segment's row indices within the combo's full
            # row sequence (seg_rows below)
            # ------------------------------------------------------------
            segments: list[tuple] = []
            for key_range, reg in entries:
                if reg["open"] is None or reg["close"] is None:
                    raise ValueError(
                        f"{combo} is missing open_position()/close_position():\n"
                        + _PATTERN
                    )
                if len(reg["dates"]) == 0:
                    # the segment's data_range selected no rows at registration
                    # (e.g. a product not yet listed in that period) — no-op
                    continue
                # min. safe guard to avoid attaching values to the wrong rows:
                # every registered date must be present in the combo's rows
                if not np.isin(reg["dates"], combo_dates).all():
                    raise ValueError(
                        f"{combo}: registered dates from data_range {key_range} "
                        "are missing from the dataframe passed to backtest()"
                    )
                # rows for the current time_segment/period
                seg_rows = np.searchsorted(combo_dates, reg["dates"])
                segments.append((reg, seg_rows))
            if not segments:
                continue
            # segments are non-overlapping date intervals → their row blocks
            # are disjoint; order by first row for the extend-forward rule
            segments.sort(key=lambda item: item[1][0])

            # ------------------------------------------------------------
            # Per-bar kernel inputs over the combo's full rows.
            # Inert defaults rule rows before the first segment (no signal →
            # no order → no position, so close params there are never read).
            # ------------------------------------------------------------
            signal_arr = np.full(num_rows, np.nan)
            order_price_arr = np.full(num_rows, np.nan)
            order_quantity_arr = np.full(num_rows, np.nan)
            first_only_arr = np.zeros(num_rows, dtype=np.bool_)
            long_only_arr = np.zeros(num_rows, dtype=np.bool_)
            short_only_arr = np.zeros(num_rows, dtype=np.bool_)
            take_profit_arr = np.full(num_rows, np.nan)
            stop_loss_arr = np.full(num_rows, np.nan)
            time_window_arr = np.full(num_rows, -1, dtype=np.int64)
            trailing_stop_arr = np.full(num_rows, np.nan)
            market_fill_at_open_arr = np.zeros(num_rows, dtype=np.bool_)
            exit_market_fill_at_open_arr = np.zeros(num_rows, dtype=np.bool_)

            for seg_idx, (reg, seg_rows) in enumerate(segments):
                # instructions: nan outside segments (the kernel skips them)
                signal_arr[seg_rows] = reg["signal"]
                _scatter_order_inputs(
                    reg["open"],
                    seg_rows,
                    combo_close,
                    order_price_arr,
                    order_quantity_arr,
                )

                # flags/risk params EXTEND FORWARD: from this segment's first
                # row until the next segment's first row (the last segment
                # extends to the end of the series). A position never survives
                # onto uncovered rows now (close-on-exit), so this only matters
                # across CONTIGUOUS segments — the gap portion is inert
                seg_start = int(seg_rows[0])
                seg_end = (
                    int(segments[seg_idx + 1][1][0])
                    if seg_idx + 1 < len(segments)
                    else num_rows
                )
                open_inputs, close_inputs = reg["open"], reg["close"]
                first_only_arr[seg_start:seg_end] = open_inputs["first_only"]
                long_only_arr[seg_start:seg_end] = open_inputs["long_only"]
                short_only_arr[seg_start:seg_end] = open_inputs["short_only"]
                # fill_price: convert 'open'/'close' to bool for numba
                # (True = fill at open)
                market_fill_at_open_arr[seg_start:seg_end] = (
                    open_inputs["fill_price"] == "open"
                )
                exit_market_fill_at_open_arr[seg_start:seg_end] = (
                    close_inputs["fill_price"] == "open"
                )
                # None → nan/-1 sentinels: a segment passing None genuinely
                # disables the feature for its stretch (distinguishable from
                # uncovered rows because extension is by segment)
                take_profit = close_inputs["take_profit"]
                stop_loss = close_inputs["stop_loss"]
                trailing_stop = close_inputs["trailing_stop"]
                time_window = close_inputs["time_window"]
                take_profit_arr[seg_start:seg_end] = (
                    np.nan if take_profit is None else float(take_profit)
                )
                stop_loss_arr[seg_start:seg_end] = (
                    np.nan if stop_loss is None else float(stop_loss)
                )
                trailing_stop_arr[seg_start:seg_end] = (
                    np.nan if trailing_stop is None else float(trailing_stop)
                )
                time_window_arr[seg_start:seg_end] = (
                    -1 if time_window is None else int(time_window)
                )

            # universe membership over the combo's rows: True on bars a
            # create_signal() segment registered, False on uncovered (gap)
            # bars. The kernel closes an open position on a False bar (the
            # product left the period's universe) and carries it across
            # contiguous covered bars. To hold across a gap, register those
            # rows (a nan signal = hold while in the universe).
            in_universe = np.zeros(num_rows, dtype=np.bool_)
            for _reg, seg_rows in segments:
                in_universe[seg_rows] = True

            combo_outs = backtest_loop_kernel(
                open_arr[pos],
                high_arr[pos],
                low_arr[pos],
                combo_close,
                volume_arr[pos],
                signal_arr,
                in_universe,
                order_price_arr,
                order_quantity_arr,
                first_only_arr,
                long_only_arr,
                short_only_arr,
                take_profit_arr,
                stop_loss_arr,
                time_window_arr,
                trailing_stop_arr,
                market_fill_at_open_arr,
                exit_market_fill_at_open_arr,
                num_rows,
            )

            signal_out[pos] = signal_arr
            for full_arr, combo_arr in zip(output_arrs, combo_outs):
                full_arr[pos] = combo_arr

        # ================================================================
        # Assign output arrays back to dataframe columns
        # ================================================================
        def _make_series(name: str, values: np.ndarray) -> nw.Series:
            return nw.new_series(name, values, nw.Float64, backend=native_backend)

        # stop_price is added unconditionally (all-nan when no stop ever
        # triggered) so the output schema never depends on the data
        result_df = df.with_columns(
            _make_series("signal", signal_out),
            _make_series("order_price", order_price_out),
            _make_series("order_size", order_size_out),
            _make_series("trade_price", trade_price_out),
            _make_series("trade_size", trade_size_out),
            _make_series("position", position_out),
            _make_series("avg_price", avg_price_out),
            _make_series("stop_price", stop_price_out),
        )
        return result_df.to_native()
