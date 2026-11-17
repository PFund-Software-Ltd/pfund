import numpy as np
from numpy.typing import NDArray

try:
    from numba import jit
except ImportError as exc:
    raise ImportError(
        "numba is required for backtesting; "
        + "install pfund with the 'core' extra, e.g. pip install pfund[core]"
    ) from exc


@jit(nopython=True, cache=True)
def portfolio_backtest_loop_kernel(
    # date-major matrices (float64[T, P]): row t = one date, column j = one product.
    price_mat: NDArray[np.float64],
    weight_mat: NDArray[np.float64],
    # universe membership (bool[T, P]): True where the product was registered
    # (in that date's universe), False otherwise. A nan weight where this is
    # True means "hold". A position whose product is False here has left the
    # universe: its committed order is still FILLED (STEP 1, no look-ahead),
    # then it is exited via a close order (STEP 3).
    in_universe: NDArray[np.bool_],
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
        1. Fill every order placed at date t-1, at its frozen price:
           close[t-1] (the prev close — same price as the product kernel's
           market fill with fill_price="close"), or 0.0 for a delisting
           write-off (see step 3). Fills are always full size: volume
           is ignored and there are no transaction costs (FAST mode is
           for prototyping; execution realism belongs to EXACT mode).
           A committed order ALWAYS fills, whatever happened to the product
           since — it LEFT THE UNIVERSE by date t, or was even DELISTED at
           t (price nan; the frozen price is its last real close). A weight
           is a committed market order, so dropping it with hindsight would
           be look-ahead bias. The position it opens is exited via a close
           order in step 3.

        ── Date t closes (close[t] and weight[t] known) ────────────────
        2. Mark to market: equity = cash + Σ_j position[j] * close[t, j].
           Equity is computed ONCE per date, before any order placement,
           so all products' targets are sized from the same number and
           rebalancing is order-independent across products.
           A held position whose product is DELISTED at t (price nan = no
           market) is worth 0.0 here, so the loss lands on the delisting
           date itself (the conservative bound modelling a delisting as a
           bankruptcy wipeout), not on the later fill of its close order.
           Record position/avg_price/cash/equity for row t.
        3. Place orders, frozen at placement and filled at t+1 (the last
           date's orders never fill, same as the product kernel):
           a. Close orders — for every position held OUTSIDE the universe
              at t: order_size = -position[j], priced at close[t, j] if the
              product still trades, or at 0.0 if it is delisted (price
              nan), which books the write-off already taken in step 2.
              The weight of an out-of-universe product is ignored.
              NOTE: the kernel records this order and its fill in the
              (T, P) outputs, but a delisted product usually has no row in
              the long df on those dates — the write-off then reaches the
              user only as the cash/equity move.
           b. Rebalance orders — for every in-universe non-nan weight[t, j].
              Weights are fractions of FREE capital — the sizing capital
              minus the value of DRIFTING positions (in-universe open
              position, nan weight on this date, i.e. "keep it"). Setting
              their value aside means explicit weights and kept positions
              compose without accidental leverage: "keep ABC + 20%/30%/50%
              in the rest" totals exactly 100% whatever ABC has drifted
              to. With no drifting positions, free capital == sizing
              capital. A position being closed in (a) is NOT set aside:
              its capital is freed on the same date the new orders fill.
                  sizing = equity[t] if compound else initial_capital
                  free = sizing - Σ_k position[k] * close[t, k]
                         over drifting products k
                  target_position = weight[t, j] * free / close[t, j]
                  order_size = target_position - position[j]
              free <= 0 on a date with a non-zero in-universe weight
              raises: the drifting positions already consume all capital,
              so weights cannot be sized (assign them explicit weights
              instead). A 0.0 weight (close) needs no sizing and never
              raises.

    Notes:
        - Inputs: in_universe must be True only where price_mat is non-nan
          (the mixin enforces this) — an in-universe nan price would turn
          the drift value, free capital and that date's orders into nan.
        - Product lifetimes: leading nan closes = listed mid-period,
          trailing nan closes = gone mid-period. A position whose product
          leaves the universe is closed (step 3a) — at the current close if
          it still trades, or written off to 0 if its data has ended; a
          position held to the GLOBAL last date while still in the universe
          is left open, same as the product kernel.
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
                # FILL — always. A placed order is a COMMITTED market order and
                # is honored at its frozen price no matter what happened to the
                # product since: left the universe, or even delisted (the frozen
                # price is its last real close). Refusing a committed fill with
                # hindsight would be look-ahead bias. STEP 3 then places a close
                # order to exit the position (at the current close, or 0 if the
                # product is now delisted).
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

        ################################################################
        ### Date t closes (close[t] and weight[t] known) ###
        ################################################################

        # ================================================================
        # STEP 2: Mark to market and record state for this date
        # ================================================================
        equity = cash
        for j in range(P):
            if position[j] != 0.0:
                c = price_mat[t, j]
                # a delisted held position (price nan) is worth 0 — its close
                # order (STEP 3) exits it at 0 — so it contributes nothing here
                if not np.isnan(c):
                    equity += position[j] * c
            position_out[t, j] = position[j]
            avg_price_out[t, j] = avg_price[j]
        cash_out[t] = cash
        equity_out[t] = equity

        # ================================================================
        # STEP 3: Place orders, to be filled at date t+1
        # ================================================================
        sizing_capital = equity if compound else initial_capital
        # Free capital: sizing capital minus the value of drifting positions —
        # an IN-UNIVERSE open position with a nan weight ("keep it"). An out-of-
        # universe held position is NOT reserved: it is being closed (close
        # order placed below) and its capital is freed next date, on the same
        # bar the new orders fill, so in-universe weights size as if it is
        # already free. (in_universe is True only on a real-price bar.)
        drift_value = 0.0
        is_rebalance = False
        for j in range(P):
            if not in_universe[t, j]:
                # weight ignored below too; a held position here is being closed
                continue
            w = weight_mat[t, j]
            if np.isnan(w):
                # nan weight and still in the universe = hold
                if position[j] != 0.0:
                    drift_value += position[j] * price_mat[t, j]
            elif w != 0.0:
                # weight 0.0 (close) needs no sizing: target is 0 regardless
                # of free capital, so it never triggers the raise below
                is_rebalance = True
        free_capital = sizing_capital - drift_value
        if is_rebalance and free_capital <= 0.0:
            raise ValueError(
                "free capital (sizing capital minus the value of drifting "
                + "positions) is non-positive on a rebalance date — weights "
                + "cannot be sized; give the drifting positions explicit "
                + "weights instead of nan, or reduce prior exposure"
            )
        for j in range(P):
            if not in_universe[t, j]:
                # out of universe: if still holding, place a close order to exit,
                # filled next date. Exit price = the current close if it still
                # trades, or 0 if the product is delisted (price nan) — a
                # bankruptcy write-off. The committed order (if any) was already
                # honored in STEP 1; this closes the resulting position.
                if position[j] != 0.0:
                    c = price_mat[t, j]
                    close_price = 0.0 if np.isnan(c) else c
                    order_price_out[t, j] = close_price
                    order_size_out[t, j] = -position[j]
                    pending_size[j] = -position[j]
                    pending_price[j] = close_price
                continue
            w = weight_mat[t, j]
            if np.isnan(w):
                continue
            c = price_mat[t, j]
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
