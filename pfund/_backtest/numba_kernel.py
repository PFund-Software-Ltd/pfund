# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
import numpy as np
from numpy.typing import NDArray
from numba import jit


@jit(nopython=True, cache=True)
def backtest_loop_kernel(
    # OHLCV arrays (float64[N])
    open_arr: NDArray[np.float64],
    high_arr: NDArray[np.float64],
    low_arr: NDArray[np.float64],
    close_arr: NDArray[np.float64],
    volume_arr: NDArray[np.float64],
    # signal from create_signal() — already computed vectorized (float64[N]: 1.0, -1.0, nan)
    signal_arr: NDArray[np.float64],
    # open_position inputs
    order_price_arr: NDArray[np.float64],    # user-specified or close prices
    order_quantity_arr: NDArray[np.float64],  # broadcast to array beforehand
    first_only: bool,
    long_only: bool,
    short_only: bool,
    # close_position inputs
    take_profit: float,        # nan = disabled
    stop_loss: float,          # nan = disabled
    time_window: int,          # -1 = disabled
    # backtest_loop inputs
    trailing_stop: float,      # nan = disabled; replaces stop_loss when set
    # fill_price controls (True = fill at open, False = fill at prev close)
    market_fill_at_open: bool,  # for entry market orders (from open_position)
    exit_market_fill_at_open: bool,  # for exit market orders (immediate stops and time_window from close_position)
    n: int,                     # number of bars (len of all input arrays)
):
    """Bar-by-bar backtest loop. Processes one bar at a time with full state tracking.

    Chronological flow within each bar i:

        ── Bar i opens (open price available) ──────────────────────────

        1. Fill pending orders placed at bar i-1 (at most ONE trade per bar).
           Orders are checked against bar i's OHLCV prices:
             - Market orders fill at open or prev_close (controlled by market_fill_at_open)
             - Limit orders fill if order_price is within [low, high].
               Gap-through: the bar's open crosses through the limit price
               (open < limit for long, open > limit for short) — always fills
               at open (the first tradeable price after the gap), ignoring
               market_fill_at_open.
             - Stop orders (SL/TP) are stop-market orders with three cases:
               a. Immediate stop: prev_close already breaches stop_price.
                  This is effectively a market order, so it fills at open or
                  prev_close (controlled by exit_market_fill_at_open).
               b. Gap-through stop: prev_close is fine but the bar's open
                  crosses through stop_price (open past the stop level).
                  Always fills at open, ignoring exit_market_fill_at_open.
               c. Non-immediate stop: stop_price is between [low, high],
                  triggered during the bar. The true market price at the
                  trigger moment is unknown with bar data, so stop_price
                  (the trigger price) is used as the fill price.

           Fill priority (highest → lowest):
             1. Immediate stop (prev_close or gap-through breaches SL/TP; SL has
                priority over TP; fills at open or prev_close, controlled by
                exit_market_fill_at_open)
             2. Time window close (fills at open or prev_close, controlled by exit_market_fill_at_open)
             3. Market order (fills at open or prev_close, controlled by market_fill_at_open)
             4. Limit order that fills before a non-immediate stop.
                Both limit and stop are on the same side of open (otherwise
                the limit would be a market order). The one closer to open
                is reached first as price travels from open toward the extreme.
                E.g. long position, price moving down from open:
                  open ──── limit_price ──── stop_price ──── low
                limit_price > stop_price → limit is hit first.
             5. Non-immediate stop (high/low breaches SL/TP during bar)
             6. Limit order (no stop contention)

        ── Bar i closes (close price available, signal[i] known) ──────

        2. Record position state for this bar. Update trailing stop's
           best_price_since_entry using bar's high (long) or low (short).

        3. Detect signal change: if signal[i] differs from the last
           forward-filled signal, reset has_traded_in_streak.

        4. Place new orders based on signal[i], to be filled at bar i+1.
           Blocked if first_only and already traded in streak.
           Note: step 3's reset ensures fills from old-streak pending orders
           don't block the new streak.

        5. Place SL/TP/trailing_stop/time_window orders based on current
           position, to be checked at bar i+1. Trailing stop replaces SL
           when set, using best_price_since_entry instead of avg_price.

    Notes:
        - Volume is ignored: trades fill at the full order size regardless of
          the bar's actual traded volume.
        - All unfilled pending orders (both regular and stop orders) are
          automatically cancelled at the end of each bar. SL/TP/trailing_stop
          are recalculated each bar since avg_price or best_price_since_entry
          may have changed (e.g. after scaling in/out).

    Returns:
        Tuple of 7 arrays (float64[N]):
            order_price_out, order_size_out,
            trade_price_out, trade_size_out,
            position_out, avg_price_out, stop_price_out
    """
    # ============================
    # Output arrays
    # ============================
    order_price_out = np.full(n, np.nan)
    order_size_out = np.full(n, np.nan)
    trade_price_out = np.full(n, np.nan)
    trade_size_out = np.full(n, np.nan)
    position_out = np.zeros(n)
    avg_price_out = np.full(n, np.nan)
    stop_price_out = np.full(n, np.nan)


    # ============================
    # Running state
    # ============================
    position = 0.0
    avg_price = np.nan
    agg_cost = 0.0
    bars_in_position = 0

    # Trailing stop state: best price seen since position entry
    best_price_since_entry = np.nan

    # Pending orders placed at end of previous bar, to be filled at current bar.
    # Exception: when fill_price is "close", orders are still filled at the current bar
    # but using prev_close (the close price of the bar where the order was placed).
    pending_order_price = np.nan
    pending_order_size = np.nan
    pending_order_side = np.nan   # +1.0 or -1.0 (signal direction of the order)
    has_pending_order = False
    # Pending stop prices (SL and TP checked separately, SL has priority per limitation #5)
    pending_sl_price = np.nan
    pending_tp_price = np.nan
    has_pending_sl = False
    has_pending_tp = False
    # Pending time_window close
    pending_tw_close = False

    # Signal tracking (forward-fill based, matching create_signal's _signal_change logic)
    last_signal_ffill = np.nan     # last non-nan signal value seen
    has_last_signal = False

    # Per-signal-streak tracking
    has_traded_in_streak = False   # for first_only: block orders after first trade in streak

    # ============================
    # Pre-computed flags for constant params (avoid np.isnan in loop)
    # ============================
    has_sl = not np.isnan(stop_loss)
    has_tp = not np.isnan(take_profit)
    has_trailing = not np.isnan(trailing_stop)
    has_tw = time_window > 0

    # ============================
    # Main loop
    # ============================
    for i in range(n):
        sig = signal_arr[i]
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]  # noqa: E741

        # ================================================================
        # STEP 1: Fill pending orders from previous bar
        # ================================================================
        # At bar i, we try to fill orders placed at end of bar i-1.
        # Only ONE trade per bar.
        # for order priority, refer to the docstring
        if i > 0:
            if has_pending_order or has_pending_sl or has_pending_tp or pending_tw_close:
                prev_c = close_arr[i - 1]
                if position > 0:
                    pos_side = 1.0
                elif position < 0:
                    pos_side = -1.0
                else:
                    pos_side = 0.0

                # --- Check SL stop conditions (highest priority) ---
                # SL direction: long position → stop below avg (triggers on price DOWN)
                #               short position → stop above avg (triggers on price UP)
                sl_immediate = False
                sl_gap = False  # True when bar gaps through SL (stop is outside this bar range)
                sl_normal = False
                sl_direction_up = False  # True if SL triggers on price going UP

                if has_pending_sl and pos_side != 0.0:
                    sl_direction_up = (pos_side == -1.0)  # short SL triggers on UP
                    if sl_direction_up:
                        sl_immediate = (prev_c >= pending_sl_price)
                        sl_gap = not sl_immediate and o > pending_sl_price
                        if not sl_immediate and not sl_gap:
                            sl_normal = (h >= pending_sl_price)
                    else:
                        sl_immediate = (prev_c <= pending_sl_price)
                        sl_gap = not sl_immediate and o < pending_sl_price
                        if not sl_immediate and not sl_gap:
                            sl_normal = (l <= pending_sl_price)

                # --- Check TP stop conditions (skip if SL already triggered) ---
                # TP direction: long position → stop above avg (triggers on price UP)
                #               short position → stop below avg (triggers on price DOWN)
                tp_immediate = False
                tp_gap = False  # True when bar gaps through TP (trigger is outside this bar range)
                tp_normal = False
                tp_direction_up = False  # True if TP triggers on price going UP

                if has_pending_tp and pos_side != 0.0 and not sl_immediate and not sl_gap and not sl_normal:
                    tp_direction_up = (pos_side == 1.0)  # long TP triggers on UP
                    if tp_direction_up:
                        tp_immediate = (prev_c >= pending_tp_price)
                        tp_gap = not tp_immediate and o > pending_tp_price
                        if not tp_immediate and not tp_gap:
                            tp_normal = (h >= pending_tp_price)
                    else:
                        tp_immediate = (prev_c <= pending_tp_price)
                        tp_gap = not tp_immediate and o < pending_tp_price
                        if not tp_immediate and not tp_gap:
                            tp_normal = (l <= pending_tp_price)

                # --- Combine stops: SL has priority over TP ---
                immediate_stop = False
                immediate_stop_price = np.nan
                immediate_stop_is_gap = False  # gap-through always fills at open
                non_immediate_stop = False
                non_immediate_stop_price = np.nan
                non_immediate_stop_direction_up = False

                if sl_immediate:
                    immediate_stop = True
                    immediate_stop_price = pending_sl_price
                elif sl_gap:
                    immediate_stop = True
                    immediate_stop_price = pending_sl_price
                    immediate_stop_is_gap = True
                elif tp_immediate:
                    immediate_stop = True
                    immediate_stop_price = pending_tp_price
                elif tp_gap:
                    immediate_stop = True
                    immediate_stop_price = pending_tp_price
                    immediate_stop_is_gap = True
                elif sl_normal:
                    non_immediate_stop = True
                    non_immediate_stop_price = pending_sl_price
                    non_immediate_stop_direction_up = sl_direction_up
                elif tp_normal:
                    non_immediate_stop = True
                    non_immediate_stop_price = pending_tp_price
                    non_immediate_stop_direction_up = tp_direction_up

                # --- Check regular order fill conditions (skip if immediate stop) ---
                is_long_order = (pending_order_side == 1.0)
                is_market = False
                is_gap_market = False  # gap-through limit always fills at open
                is_limit = False

                if has_pending_order and not immediate_stop:
                    # Market order: order price is already aggressive vs prev_close
                    # Long market: prev_close <= order_price (buying at or above market)
                    # Short market: prev_close >= order_price (selling at or below market)
                    if is_long_order:
                        is_market = (prev_c <= pending_order_price)
                        is_gap_market = not is_market and o < pending_order_price
                    else:
                        is_market = (prev_c >= pending_order_price)
                        is_gap_market = not is_market and o > pending_order_price
                    is_market = is_market or is_gap_market

                    # Limit order: order price within bar's [low, high] range.
                    # One-sided check suffices: the other bound is guaranteed by is_market check.
                    if not is_market:
                        if is_long_order:
                            is_limit = (pending_order_price >= l)
                        else:
                            is_limit = (pending_order_price <= h)

                # --- Check if limit order fills before non-immediate stop ---
                # When both a limit order and a non-immediate stop could trigger in the same bar,
                # the limit fills first if its price is "on the way" to the stop:
                #   - Price going DOWN to stop: long limit at higher price → reached first
                #   - Price going UP to stop: short limit at lower price → reached first
                limit_before_stop = False
                if is_limit and non_immediate_stop:
                    if non_immediate_stop_direction_up:
                        # Price going up to stop; short limit (lower price) fills first
                        limit_before_stop = (not is_long_order) and (pending_order_price <= non_immediate_stop_price)
                    else:
                        # Price going down to stop; long limit (higher price) fills first
                        limit_before_stop = is_long_order and (pending_order_price >= non_immediate_stop_price)

                # --- Determine trade price and size by priority ---
                trade_price = np.nan
                trade_size = np.nan
                has_fill = False
                is_position_close = False

                if immediate_stop:
                    # Priority 1: immediately triggered stop
                    # Gap-through: always fill at open (prev_close didn't breach the stop)
                    # Regular immediate: fill per exit_market_fill_at_open policy
                    trade_price = o if (immediate_stop_is_gap or exit_market_fill_at_open) else prev_c
                    trade_size = -position
                    has_fill = True
                    is_position_close = True
                    stop_price_out[i - 1] = immediate_stop_price  # i - 1 means it's in the order row, which is consistent with vectorized backtesting
                elif pending_tw_close:
                    # Priority 2: time window close
                    trade_price = o if exit_market_fill_at_open else prev_c
                    trade_size = -position
                    has_fill = True
                    is_position_close = True
                elif is_market:
                    # Priority 3: market order
                    # Gap-through: always fill at open (limit price outside bar range)
                    # Regular market: fill per market_fill_at_open policy
                    trade_price = o if (is_gap_market or market_fill_at_open) else prev_c
                    trade_size = pending_order_size
                    has_fill = True
                elif limit_before_stop:
                    # Priority 4: limit order (fills before non-immediate stop)
                    trade_price = pending_order_price
                    trade_size = pending_order_size
                    has_fill = True
                elif non_immediate_stop:
                    # Priority 5: non-immediately triggered stop
                    trade_price = non_immediate_stop_price
                    trade_size = -position
                    has_fill = True
                    is_position_close = True
                    stop_price_out[i - 1] = non_immediate_stop_price  # i - 1 means it's in the order row, which is consistent with vectorized backtesting
                elif is_limit:
                    # Priority 6: limit order (no stop contention)
                    trade_price = pending_order_price
                    trade_size = pending_order_size
                    has_fill = True

                # --- Execute trade ---
                if has_fill:
                    trade_price_out[i] = trade_price
                    trade_size_out[i] = trade_size

                    if is_position_close:
                        # Close entire position (SL/TP/TW only — guaranteed trade_size = -position)
                        position = 0.0
                        avg_price = np.nan
                        agg_cost = 0.0
                        bars_in_position = 0
                        best_price_since_entry = np.nan
                    else:
                        # Regular trade (market/limit from signal): handles open, add, reduce,
                        # close-to-flat, and flip. An opposite-signal order that happens to
                        # close the position (new_position == 0) is NOT treated as
                        # is_position_close — resets are handled by the general logic below.
                        if position > 0:
                            old_pos_sign = 1.0
                        elif position < 0:
                            old_pos_sign = -1.0
                        else:
                            old_pos_sign = 0.0

                        new_position = position + trade_size
                        if new_position > 0:
                            new_pos_sign = 1.0
                        elif new_position < 0:
                            new_pos_sign = -1.0
                        else:
                            new_pos_sign = 0.0

                        is_sign_diff = new_pos_sign != old_pos_sign

                        # Update aggregate cost for avg_price calculation
                        if position == 0.0:
                            # Opening from flat: fresh start
                            agg_cost = trade_price * trade_size
                        elif new_position == 0.0:
                            # Closed to flat
                            agg_cost = 0.0
                        elif is_sign_diff:
                            # Position flipped: new streak starts with the remainder
                            # e.g. position=+2, trade=-3 → new_position=-1
                            # Only the -1 part contributes to the new avg_price
                            agg_cost = trade_price * new_position
                        else:
                            # Same side after trade — distinguish adding vs reducing
                            is_adding = (trade_size > 0.0 and position > 0.0) or (trade_size < 0.0 and position < 0.0)
                            if is_adding:
                                # Adding to existing position
                                agg_cost += trade_price * trade_size
                            else:
                                # Partial reduction: remaining position keeps prior avg_price
                                agg_cost = avg_price * new_position

                        # Reset bars_in_position if position side changed
                        if is_sign_diff:
                            bars_in_position = 0

                        position = new_position
                        if position != 0.0:
                            avg_price = agg_cost / position
                        else:
                            avg_price = np.nan
                            agg_cost = 0.0
                            best_price_since_entry = np.nan

                        # Reset best_price on new position (from flat or side flip)
                        if position != 0.0 and is_sign_diff:
                            best_price_since_entry = trade_price

                        # Mark that a trade occurred (only for signal-driven trades,
                        # not automatic closes like SL/TP/TW). If signal changes at
                        # this bar, step 3 will reset this before order placement.
                        has_traded_in_streak = True

            # Clear all pending orders (whether filled or not — unfilled orders are cancelled)
            pending_order_price = np.nan
            pending_order_size = np.nan
            pending_order_side = np.nan
            has_pending_order = False
            pending_sl_price = np.nan
            pending_tp_price = np.nan
            has_pending_sl = False
            has_pending_tp = False
            pending_tw_close = False

        
        
        ################################################################
        ### Bar i closes (close price available, signal[i] known) ###
        ################################################################
        
        
        
        # ================================================================
        # STEP 2: Record position state for this bar
        # ================================================================
        position_out[i] = position
        avg_price_out[i] = avg_price
        if position != 0.0:
            bars_in_position += 1
            # Update trailing stop's best price seen since entry
            if position > 0.0:
                if h > best_price_since_entry:
                    best_price_since_entry = h
            else:
                if l < best_price_since_entry:
                    best_price_since_entry = l


        # ================================================================
        # STEP 3+4: Detect signal change and place new orders
        # ================================================================
        # Forward-fill signal: if current signal is non-nan and differs from
        # the last forward-filled value, it's a new signal streak.
        # Orders placed at end of bar i, to be filled at bar i+1.
        if not np.isnan(sig):
            if not has_last_signal or sig != last_signal_ffill:
                last_signal_ffill = sig
                has_last_signal = True
                # Reset per-streak state
                has_traded_in_streak = False
            # Check if we can place an order
            can_order = True
            if first_only and has_traded_in_streak:
                # first_only: only one trade per signal streak
                can_order = False

            if can_order:
                order_px = order_price_arr[i]
                order_qty = order_quantity_arr[i]

                has_order = False
                order_size = np.nan  # only read when has_order is True
                if long_only and sig == -1.0:
                    # Long-only close: sell exactly the current long position
                    if position > 0.0:
                        order_size = -position
                        has_order = True
                elif short_only and sig == 1.0:
                    # Short-only close: buy exactly the current short position
                    if position < 0.0:
                        order_size = -position
                        has_order = True
                else:
                    # Normal order: signal * quantity (skip if NaN price or qty → no order on this bar)
                    if not np.isnan(order_px) and not np.isnan(order_qty):
                        order_size = sig * order_qty
                        # Add offset to close existing opposite position before opening new
                        # e.g. position=+2, signal=-1, qty=1 → order_size = -1 + (-2) = -3
                        if position != 0.0:
                            if (position > 0.0 and sig < 0.0) or (position < 0.0 and sig > 0.0):
                                order_size += -position
                        has_order = True

                if has_order:
                    # Record the order output
                    # order_price = abs(signal) * user_order_price (matches vectorized)
                    order_price_out[i] = order_px
                    order_size_out[i] = order_size
                    # Store as pending for next bar
                    pending_order_price = order_px
                    pending_order_size = order_size
                    pending_order_side = sig
                    has_pending_order = True

        # ================================================================
        # STEP 5: Place stop / time_window orders based on current position
        # ================================================================
        # Stop/TW orders placed at end of bar i, checked at bar i+1.
        if position != 0.0:
            pos_side = 1.0 if position > 0.0 else -1.0

            # SL stop price: move against position direction
            # Long: avg_price * (1 - stop_loss)  → below avg_price
            # Short: avg_price * (1 + stop_loss) → above avg_price
            if has_sl:
                pending_sl_price = avg_price * (1.0 - pos_side * stop_loss)
                has_pending_sl = True

            # Trailing stop: replaces SL when set — uses best price instead of avg_price
            # Long: best_high * (1 - trailing_stop)  → trails below the peak
            # Short: best_low * (1 + trailing_stop)  → trails above the trough
            if has_trailing:
                pending_sl_price = best_price_since_entry * (1.0 - pos_side * trailing_stop)
                has_pending_sl = True

            # TP stop price: move with position direction
            # Long: avg_price * (1 + take_profit)  → above avg_price
            # Short: avg_price * (1 - take_profit) → below avg_price
            if has_tp:
                pending_tp_price = avg_price * (1.0 + pos_side * take_profit)
                has_pending_tp = True

            # Time window: close position when held for 'time_window' bars
            # SL/TP may also be pending; priority is resolved in step 1 of the next bar
            # (immediate_stop > tw_close > market > limit_before_stop > non_immediate_stop)
            if has_tw and bars_in_position >= time_window:
                pending_tw_close = True

    return (
        order_price_out, order_size_out,
        trade_price_out, trade_size_out,
        position_out, avg_price_out, stop_price_out,
    )
