# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
import numpy as np
from numba import jit
from numpy.typing import NDArray


@jit(nopython=True, cache=True)
def portfolio_backtest_loop_kernel(
    # date-major matrices (float64[T, P]): row t = one date, column j = one product.
    # Pivoted from the long df by PortfolioBacktestMixin.backtest();
    # nan close = product has no bar that date. Missingness is validated
    # upstream to be a prefix (listed mid-period) and/or a suffix
    # (delisted/halted mid-period); mid-series gaps are rejected.
    price_mat: NDArray[np.float64],
    # weight from create_weight(): signed fraction of FREE capital — sizing
    # capital minus the value of drifting positions (see STEP 4).
    # nan = hold (no instruction, weight drifts), 0.0 = close the position.
    # Magnitude is never checked: |w| > 1 is leverage, that's the user's business.
    weight_mat: NDArray[np.float64],
    # backtest() inputs
    initial_capital: float,
    compound: bool,  # True: size on current equity; False: size on initial_capital
    T: int,  # number of dates
    P: int,  # number of products
):
    """Date-by-date portfolio rebalancing loop over one shared equity pot.

    Row/fill convention deliberately mirrors the product kernel so both
    backtest modes share one output schema and row alignment (they feed the
    same analysis engine):

        ── Date t opens ────────────────────────────────────────────────
        1. Fill rebalance orders placed at date t-1, at close[t-1] (the
           prev close — same price as the product kernel's market fill
           with fill_price="close"). Fills are always full size: volume
           is ignored and there are no transaction costs (FAST mode is
           for prototyping; execution realism belongs to EXACT mode).
           A pending order of a product whose data has ended (t is past
           its last bar) is CANCELLED instead — there is no market to
           fill in (the per-product version of "the last date's orders
           never fill").
        2. Force-close any position whose product's data ended at t-1
           (delisted/bankrupt/halted): the position is converted to cash
           at the product's last close and zeroed. Without this step,
           mark-to-market (which skips nan closes) would silently drop
           the position's value from equity.

        ── Date t closes (close[t] and weight[t] known) ────────────────
        3. Mark to market: equity = cash + Σ_j position[j] * close[t, j].
           Equity is computed ONCE per date, before any order placement,
           so all products' targets are sized from the same number and
           rebalancing is order-independent across products.
           Record position/avg_price/cash/equity for row t.
        4. Place rebalance orders for every non-nan weight[t, j].
           Weights are fractions of FREE capital — the sizing capital
           minus the value of DRIFTING positions (open position, nan
           weight on this date, i.e. "keep it"). Setting their value
           aside means explicit weights and kept positions compose
           without accidental leverage: "keep ABC + 20%/30%/50% in the
           rest" totals exactly 100% whatever ABC has drifted to. With
           no drifting positions, free capital == sizing capital.
               sizing = equity[t] if compound else initial_capital
               free = sizing - Σ_k position[k] * close[t, k]
                      over drifting products k
               target_position = weight[t, j] * free / close[t, j]
               order_size = target_position - position[j]
           free <= 0 on a date with weight instructions raises: the
           drifting positions already consume all capital, so weights
           cannot be sized (assign them explicit weights instead).
           Order size/price are frozen at placement and filled at t+1.
           Last date's orders never fill (same as the product kernel).

    Notes:
        - Product lifetimes: leading nan closes = listed mid-period,
          trailing nan closes = gone mid-period. A position held when its
          product's data ends is force-closed at the last close (step 2);
          a position held to the GLOBAL last date is left open, same as
          the product kernel.
        - No exposure/leverage checks: explicit weights summing above 1
          lever the free capital, cash may go negative (borrowing),
          equity may go negative — the kernel computes, the user judges.
          The only refusal is the free <= 0 raise above, where sizing
          itself is undefined.
        - No stops / take-profit / time-window / limit prices: a portfolio
          is about rebalancing, not taking profit or stopping loss.

    Returns:
        Tuple of 6 matrices (float64[T, P]) + 2 arrays (float64[T]):
            order_price_out, order_size_out,
            trade_price_out, trade_size_out,
            position_out, avg_price_out,
            cash_out, equity_out
    """
    # ============================
    # Output arrays
    # ============================
    order_price_out = np.full((T, P), np.nan)
    order_size_out = np.full((T, P), np.nan)
    trade_price_out = np.full((T, P), np.nan)
    trade_size_out = np.full((T, P), np.nan)
    position_out = np.zeros((T, P))
    avg_price_out = np.full((T, P), np.nan)
    cash_out = np.full(T, np.nan)
    equity_out = np.full(T, np.nan)

    # ============================
    # Running state
    # ============================
    position = np.zeros(P)
    avg_price = np.full(P, np.nan)
    agg_cost = np.zeros(P)
    cash = initial_capital

    # Pending rebalance orders placed at end of previous date (nan = none).
    pending_size = np.full(P, np.nan)
    pending_price = np.full(P, np.nan)

    # ============================
    # Per-product end of life: last bar with a real close. Upstream
    # validation guarantees no mid-series gaps, so every t > last_bar[j]
    # means the product is gone (delisted/halted); nans before the first
    # bar (listed mid-period) need no index — weights can't exist there
    # and mark-to-market skips nan closes anyway.
    # ============================
    last_bar = np.full(P, -1, dtype=np.int64)
    for j in range(P):
        for t in range(T):
            if not np.isnan(price_mat[t, j]):
                last_bar[j] = t

    # ============================
    # Main loop
    # ============================
    for t in range(T):
        # ================================================================
        # STEP 1: Fill pending orders from previous date
        # ================================================================
        if t > 0:
            for j in range(P):
                size = pending_size[j]
                if np.isnan(size):
                    continue
                if t > last_bar[j]:
                    # the product's data ended before the fill date: there is
                    # no market to fill in — cancel (the per-product version
                    # of "the last date's orders never fill")
                    pending_size[j] = np.nan
                    pending_price[j] = np.nan
                    continue
                price = pending_price[j]
                trade_price_out[t, j] = price
                trade_size_out[t, j] = size
                cash -= size * price

                # avg_price bookkeeping (same semantics as the product kernel)
                old_pos = position[j]
                new_pos = old_pos + size
                if old_pos == 0.0:
                    # Opening from flat: fresh start
                    agg_cost[j] = price * size
                elif new_pos == 0.0:
                    # Closed to flat
                    agg_cost[j] = 0.0
                elif (new_pos > 0.0) != (old_pos > 0.0):
                    # Position flipped: new streak starts with the remainder
                    agg_cost[j] = price * new_pos
                elif (size > 0.0) == (old_pos > 0.0):
                    # Adding to existing position
                    agg_cost[j] += price * size
                else:
                    # Partial reduction: remaining position keeps prior avg_price
                    agg_cost[j] = avg_price[j] * new_pos

                position[j] = new_pos
                if new_pos != 0.0:
                    avg_price[j] = agg_cost[j] / new_pos
                else:
                    avg_price[j] = np.nan
                    agg_cost[j] = 0.0

                pending_size[j] = np.nan
                pending_price[j] = np.nan

        # ================================================================
        # STEP 2: Force-close positions whose product's data has ended
        # ================================================================
        # t == last_bar[j] + 1 is the first date with no price for j; from
        # here on STEP 3's mark-to-market skips j (nan close), so without
        # this step the position's value would silently vanish from equity.
        for j in range(P):
            if position[j] != 0.0 and t == last_bar[j] + 1:
                # TBD: settlement price — price data alone cannot distinguish
                # bankruptcy (recovery ~0) from a halt/migration (recovery
                # ~last price). Chosen: the LAST CLOSE, the last observable
                # mark — it captures every loss already in the price series
                # and keeps equity continuous. Alternative: settle at 0.0
                # (full write-off — overstates the loss for non-bankrupt
                # delistings, and is a windfall for shorts).
                settlement_price = price_mat[last_bar[j], j]
                # TBD: this (t, j) cell has no row in the long df (no bar →
                # no row), so the forced close cannot surface in the result
                # df's trade columns; it is still recorded in the panel for
                # internal consistency, and reaches the user as the
                # cash/equity jump on this date.
                trade_price_out[t, j] = settlement_price
                trade_size_out[t, j] = -position[j]
                cash += position[j] * settlement_price
                position[j] = 0.0
                avg_price[j] = np.nan
                agg_cost[j] = 0.0

        ################################################################
        ### Date t closes (close[t] and weight[t] known) ###
        ################################################################

        # ================================================================
        # STEP 3: Mark to market and record state for this date
        # ================================================================
        equity = cash
        for j in range(P):
            if position[j] != 0.0:
                c = price_mat[t, j]
                if not np.isnan(c):
                    equity += position[j] * c
        cash_out[t] = cash
        equity_out[t] = equity
        for j in range(P):
            position_out[t, j] = position[j]
            avg_price_out[t, j] = avg_price[j]

        # ================================================================
        # STEP 4: Place rebalance orders, to be filled at date t+1
        # ================================================================
        sizing_capital = equity if compound else initial_capital
        # Free capital: sizing capital minus the value of drifting positions
        # (open position, nan weight = "keep it"). A position can only exist
        # within its product's bar life, so its close here is never nan
        # (force-closed in STEP 2 once the data ends) — checked defensively.
        drift_value = 0.0
        has_instruction = False
        for j in range(P):
            if np.isnan(weight_mat[t, j]):
                if position[j] != 0.0 and not np.isnan(price_mat[t, j]):
                    drift_value += position[j] * price_mat[t, j]
            elif weight_mat[t, j] != 0.0 and not np.isnan(price_mat[t, j]):
                # weight 0.0 (close) needs no sizing: target is 0 regardless
                # of free capital, so it never triggers the raise below
                has_instruction = True
        free_capital = sizing_capital - drift_value
        if has_instruction and free_capital <= 0.0:
            raise ValueError(
                "free capital (sizing capital minus the value of drifting "
                + "positions) is non-positive on a rebalance date — weights "
                + "cannot be sized; give the drifting positions explicit "
                + "weights instead of nan, or reduce prior exposure"
            )
        for j in range(P):
            w = weight_mat[t, j]
            if np.isnan(w):
                continue
            c = price_mat[t, j]
            if np.isnan(c):
                # no bar for this product on this date (prefix missingness);
                # weights here are rejected upstream — skip defensively
                continue
            target_position = w * free_capital / c
            delta = target_position - position[j]
            if delta != 0.0:
                order_price_out[t, j] = c
                order_size_out[t, j] = delta
                pending_size[j] = delta
                pending_price[j] = c

    return (
        order_price_out,
        order_size_out,
        trade_price_out,
        trade_size_out,
        position_out,
        avg_price_out,
        cash_out,
        equity_out,
    )
