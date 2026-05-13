"""
Portfolio Backtesting Design Notes (Brainstorm — Not Implemented)
================================================================

Goal: Extend vectorized backtesting to support multiple products (portfolio-level).

Core Idea
---------
- Vstack multiple products into a single DataFrame.
  All products must share the same resolution (bar data, no tick data)
  so rows align naturally by timestamp.
- Add a `product` column to identify each instrument.
- Replace binary signals (+1/-1) with continuous weights: -1 <= w <= +1,
  where the weight represents the target allocation for each product.

Data Layout
-----------
    timestamp | product | open | high | low | close | signal (weight)
    --------- | ------- | ---- | ---- | --- | ----- | ---------------
    2024-01-01| AAPL    | ...  | ...  | ... | ...   | 0.3
    2024-01-01| GOOGL   | ...  | ...  | ... | ...   | -0.2
    2024-01-02| AAPL    | ...  | ...  | ... | ...   | 0.5
    2024-01-02| GOOGL   | ...  | ...  | ... | ...   | -0.5

Key Mechanical Changes
----------------------
Every stateful operation (shift, cumsum, ffill, diff) must be scoped per product:

- Pandas: groupby('product').shift(1), or use MultiIndex on (timestamp, product)
- Polars: .shift(1).over('product')  — cleaner, no MultiIndex needed.
  Optionally use over('product', order_by='timestamp') for explicit ordering
  instead of relying on physical row order.

This affects ~15 shift/cumsum/ffill/diff calls across open_position() and close_position().
The fill logic, SL/TP logic, and streak detection all stay the same algorithm,
just scoped per product.

Weight interpretation: signal column holds a float weight instead of +1/-1.
order_quantity = abs(weight) * capital_per_product (or similar scaling).

What Works in Vectorized Mode (Independent Weights)
----------------------------------------------------
- Weights are predetermined per (timestamp, product) — no cross-product dependency.
- No portfolio-level capital constraint (or assume infinite leverage).
- SL/TP is per-product, independent.
- Each product is essentially an independent backtest sharing a DataFrame.
- This is the "embarrassingly parallel" case.

What Requires Hybrid/Event-Driven Mode
---------------------------------------
- Cross-product capital constraints (total weight <= 1.0 or leverage limit).
- Rebalancing where weights are relative to each other (groupby timestamp normalization).
- Cash tracking: buying product A reduces capital available for product B.
- Portfolio-level risk management.
- SL/TP on one product freeing capital for another — sequential logic, not vectorizable.

Open Design Decision
--------------------
NOT YET DECIDED: whether to modify the existing BacktestDataFrame classes in
pandas.py / polars.py to be product-aware (e.g. detect a 'product' column and
auto-scope operations), or create separate PortfolioBacktestDataFrame subclasses
in a new file.

Trade-offs:
- Modifying existing: single code path, no duplication, but adds complexity
  to already dense logic (every shift/cumsum gets conditional .over('product')).
- New subclass: cleaner separation of concerns, but duplicates the core
  fill/SL/TP algorithm. Could share via a mixin or by calling super() with
  pre-grouped data.

Summary
-------
    Simple portfolio (vectorized)     | Full portfolio (hybrid/event-driven)
    --------------------------------- | ------------------------------------
    Independent weights per product   | Cross-product capital constraints
    No capital tracking               | Rebalancing with normalization
    Per-product SL/TP                 | Portfolio-level risk management
    groupby(product) / .over(product) | Sequential loop per timestamp


Interesting Equivalence: pfund backtesting ≈ Portfolio Backtesting
-----------------------------------------------------------------
When you strip pfund's vectorized backtest down to its simplest form:
    - Multi-product, same resolution bar data
    - Signals (+1/-1) per product converted to weights (-1 <= w <= +1)
    - Fill at close price (market orders only)
    - No SL/TP, no limit orders, no time_window

Then pfund's vectorized backtest is conceptually equivalent to what
dedicated portfolio backtesting libraries (e.g. bt, vectorbt) compute:

    portfolio P&L = sum(weight_i * return_i) per bar

Both approaches reduce to the same per-bar weighted return calculation.
The weights-based approach is more elegant because weight changes *implicitly*
define trades — there is no explicit order/fill model needed.

The one remaining subtle difference is compounding:

    Portfolio libraries:  portfolio_value_{t+1} = portfolio_value_t * (1 + sum(w_i * r_i))
                          (geometric, weights are relative to current portfolio value)

    pfund vectorized:     pnl = trade_size * (exit_price - entry_price)
                          (arithmetic, fixed quantities)

    → Over many bars these diverge. To fully close the gap, pfund would need
      to scale order_quantity proportionally to current portfolio value at each
      rebalancing bar, which introduces a sequential dependency and effectively
      pushes it toward hybrid mode.

Why this is interesting:
    Portfolio backtesting libraries give up trade-level realism (exact fill prices,
    limit orders, SL/TP, gap handling) in exchange for an elegant closed-form
    weighted-return model. pfund's vectorized mode is the same thing underneath —
    the fill logic, SL/TP, and order types are what distinguish it and justify its
    additional complexity. Without those features, a weights-based approach is
    simpler and equally correct.
"""
