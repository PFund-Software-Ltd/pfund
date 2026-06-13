# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportReturnType=false, reportMissingTypeArgument=false, reportImplicitStringConcatenation=false, reportUnknownParameterType=false, reportArgumentType=false, reportCallIssue=false
import datetime

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT, IntoSeries
from pfund_kit.style import RichColor, TextStyle, cprint

from pfund._backtest.portfolio_backtest_kernel import portfolio_backtest_loop_kernel
from pfund._backtest.product_backtest_mixin import (
    _group_positions,
    _resolve_targets,
    _series_to_positional_float64,
    _to_float64,
)

# one entry per create_weight() call
_registry: dict[tuple, dict] = {}
# ((start_date, end_date), "BYBIT_BTC_USDT_PERPETUAL"): {  # key = (data_range, product); range resolved
#   "weight": np.array([nan, ..., 0.5]) — positional over the entry's target rows
#   "dates":  np.array([date1, ...]) — target rows' dates, pin the exact rows so
#             backtest() can verify it scatters weights onto the same rows
# }

_PATTERN = (
    "stride = '1d'  # rebalance period: '1d', '1w', '1mo', ...\n"
    "periods = sorted(df['date'].dt.truncate(stride).unique().to_list())\n"
    "for period in periods:\n"
    "    # expanding point-in-time window: history up to the current period\n"
    "    periodic_df = df.filter(pl.col('date').dt.truncate(stride) <= period)\n"
    "    universe = sorted(periodic_df['product'].unique().to_list())\n"
    "    # NOTE: add custom screening logic to shortlist the universe\n"
    "    for product in universe:\n"
    "        product_df = periodic_df.filter(pl.col('product') == product)\n"
    "        product_df.create_weight(weight=..., data_range=(period_start, period_end))\n"
    "\n"
    "df.backtest(initial_capital=1_000_000, compound=True)"
)


def _clear_registry() -> None:
    _registry.clear()


def _get_registry_key(
    df: nw.DataFrame, data_range: tuple[datetime.date, datetime.date] | None
) -> tuple[tuple[datetime.date, datetime.date], str]:
    """Build the registry key ((start_date, end_date), product) for one
    create_weight() call — both halves derived from the call: the product from
    the df (which must hold exactly one), the range as passed (None → the df's
    first/last dates).
    """
    products = df.get_column("product").unique().to_list()
    if len(products) != 1:
        raise ValueError(
            "create_weight() must be called on a df holding exactly one product:\n"
            + _PATTERN
        )
    product = products[0]
    if data_range is None:
        date_col = df.get_column("date")
        data_range = (date_col.item(0), date_col.item(len(date_col) - 1))
    return (data_range, product)


class PortfolioBacktestMixin:
    def create_weight(
        self: IntoDataFrameT,
        weight: IntoSeries,
        data_range: tuple[datetime.date, datetime.date] | None = None,
    ) -> IntoDataFrameT:
        """Registers weights for this product's rows within data_range.

        A weight is the signed fraction of the portfolio's FREE capital to
        hold in this product. Free capital is the sizing capital (current
        equity, or initial_capital when compound=False) minus the value of
        DRIFTING positions — products holding a position with no weight
        instruction (nan) on that date:
            weight = 0.1  → hold long worth 10% of free capital
            weight = -0.5 → hold short worth 50% of free capital
            weight = 0.0  → close the position
            weight = nan  → no instruction, keep the position: its weight
                            drifts and its value is set aside from free capital
        Why free capital: "keep ABC (nan) and put 20%/30%/50% in the rest"
        then totals exactly 100% of capital whatever ABC has drifted to —
        explicit weights and kept positions compose without accidental
        leverage, and you never need to know a drifted weight to size the
        rest. With no drifting positions, free capital equals sizing capital.
        Every non-nan weight is a rebalance instruction toward the target
        position (target = weight * free_capital / close, the delta is traded).
        Magnitudes and per-date weight sums are never altered — |weight| > 1 or
        explicit weights summing above 1 simply lever the free capital, that's
        the user's backtest (backtest(check=True) warns on the latter);
        only inf/-inf is rejected.

        Pure REGISTRATION step: stores the weights (with their rows' dates) in
        the session registry — no column is written to any df and self is
        returned unchanged. The weights appear as a column in the result df
        produced by backtest(); rows never covered by any call stay nan
        (no instruction → drift). Must be called on a df holding exactly one
        product.

        Registration is point-in-time: a row's weight can only be registered
        once — a data_range that overlaps an already-registered range of the
        same product is an error (the past is never rewritten).

        Args:
            weight: a native series (pandas or polars) matching the
                dataframe's backend — positional weights over the SAME df
                create_weight() was called on (the full expanding window), so
                its length must match the df exactly; it is sliced internally to
                the data_range rows (the rows actually registered). Each row is
                spelled out explicitly: a non-nan value rebalances toward that
                target, nan means no instruction (drift/hold). To rebalance once
                at the period's end, pass a series that is nan everywhere except
                the period's last (data_range) row.
            data_range: (start_date, end_date), both inclusive — the rows this
                call covers. A range selecting no rows (e.g. a delisted
                product still in the universe) is a no-op.
                None (default) → configure over ALL rows of this df (full
                span). Registration is still register-once: this raises if it
                would overlap an already-registered range of the product.
        """
        df = nw.from_native(self)
        key = _get_registry_key(df, data_range)
        data_range, product = key
        target_pos, target_dates = _resolve_targets(
            _registry,
            product,
            lambda k: k[1],
            df,
            data_range,
        )

        # weight is positional over the SAME df create_weight() was called on
        # (the full expanding window); validate that length, then keep only the
        # data_range rows — mirrors create_signal's signal_arr[target_pos], so
        # every series in the chain follows one rule (positional over the df)
        weight_full = _series_to_positional_float64(weight, len(df), "weight")
        weight_arr = weight_full[target_pos]

        # register weight series and its target dates
        _registry[key] = {
            "dates": target_dates,
            "weight": weight_arr,
        }

        # dynamically add backtest method to the dataframe class
        df_class = type(self)
        df_class.backtest = PortfolioBacktestMixin.backtest
        return self

    def backtest(
        self: IntoDataFrameT,
        initial_capital: float = 1_000_000,
        compound: bool = True,
        check: bool = True,
    ) -> IntoDataFrameT:
        """Runs the portfolio rebalancing backtest over all registered products.

        One shared equity pot; every non-nan weight rebalances its product toward
        target = weight * free_capital / close, where free capital is the sizing
        capital minus the value of drifting positions (open position, nan weight
        — "keep it") on that date; see create_weight(). Orders are placed at each
        date's close (order_price = close) and filled at the NEXT date at that
        same close price — the same row/fill convention as product backtesting
        (both feed the same analysis engine); the last date's orders never fill.
        Volume never caps fills and there are no transaction costs (FAST mode
        is for prototyping; execution realism belongs to EXACT mode).

        Raises if free capital is non-positive on a date with weight
        instructions (drifting positions already consume all capital, so
        weights cannot be sized — give them explicit weights instead of nan).

        Args:
            initial_capital: starting cash of the portfolio.
            compound: True → sizing capital is the CURRENT equity (mark-to-market,
                computed once per date before any order placement);
                False → sizing capital is always initial_capital.
            check: warn when the positive weights on any date sum above 100% —
                under free-capital sizing that is either a typo or deliberate
                leverage. Pure validation: results are never altered;
                pass False to silence intentional leverage.

        Result columns (weight nan = no instruction for that row — drift;
        all columns nan = the row's product was never configured):
            weight, order_price, order_size, trade_price, trade_size,
            position, avg_price — per (date, product) row;
            cash, equity — portfolio-level, broadcast per date.
        """
        if not _registry:
            raise ValueError(
                "backtest() cannot run, setup should have been called properly:\n"
                + _PATTERN
            )
        if not initial_capital > 0:
            raise ValueError("'initial_capital' must be positive")

        df = nw.from_native(self)
        native_backend = nw.get_native_namespace(df)
        n = len(df)

        resolutions = df.get_column("resolution").unique().to_list()
        if len(resolutions) > 1:
            raise ValueError(
                f"portfolio backtesting requires a single resolution, got {sorted(resolutions)} — "
                "mixing resolutions is not supported; resample your data to one resolution first"
            )

        close_arr = _to_float64(df.get_column("close"))
        date_arr = df.get_column("date").to_numpy()

        # Group rows by product once (O(n log n))
        positions_by_product = _group_positions(
            [df.get_column("product").to_numpy()], n
        )

        # ================================================================
        # Regroup the flat registry into {product -> [(date_range, reg), ...]},
        # bucketing every segment (one create_weight() call) under its product:
        #   _registry = {
        #       (jan, "AAPL"): reg1,  # reg = {"dates": [...], "weight": [...]}
        #       (feb, "AAPL"): reg2,
        #       (jan, "TSLA"): reg3,
        #   }
        #   entries_by_product = {
        #       "AAPL": [(jan, reg1), (feb, reg2)],   # 2 segments
        #       "TSLA": [(jan, reg3)],                # 1 segment
        #   }
        # ================================================================
        entries_by_product: dict[str, list[tuple]] = {}
        for (key_range, product), reg in _registry.items():
            entries_by_product.setdefault(product, []).append((key_range, reg))

        # ================================================================
        # Per-product row positions (products in first-registration order)
        # ================================================================
        products: list[str] = list(entries_by_product)
        P = len(products)
        product_positions: list[np.ndarray] = []
        for product in products:
            # positions of this product's rows (ascending), computed once above
            pos: np.ndarray | None = positions_by_product.get((product,))
            if pos is None:
                raise ValueError(
                    f"{product} has no rows in the dataframe passed to backtest()"
                )
            product_positions.append(pos)

        # ================================================================
        # Pivot long df → date-major (T, P) matrices for the kernel.
        # Rows = union_dates (T), columns = products (P); a missing
        # (date, product) row stays nan. e.g. for AAPL/MSFT where MSFT
        # halts a day early:
        #
        #              AAPL  MSFT
        #     01-01 │  100   200
        #     01-02 │  101   201
        #     01-03 │  102   nan   ← MSFT has no row → stays nan
        #            └──────────── price_mat (weight_mat filled the same way)
        # ================================================================
        union_dates = np.unique(
            np.concatenate([date_arr[pos] for pos in product_positions])
        )
        T = len(union_dates)
        price_mat = np.full((T, P), np.nan)
        weight_mat = np.full((T, P), np.nan)
        # universe membership: True where a create_weight() call covered this
        # (date, product) cell, False elsewhere. Set alongside weight_mat in the
        # scatter loop below. A nan weight where this is True means "hold" (in
        # universe, no instruction); a cell that is False sits outside every
        # registered range — the product is not in that date's universe. This is
        # the bit weight_mat alone cannot carry: covered-but-nan and uncovered
        # both read as nan in weight_mat, but differ here (True vs False).
        in_universe = np.full((T, P), False)
        # each product's row positions in the (T, P) panel; reused to scatter
        # outputs back, so round-trip row alignment is structural
        t_idxs: list[np.ndarray] = []
        for j, product in enumerate(products):
            pos = product_positions[j]
            product_dates = date_arr[pos]
            if (
                len(product_dates) > 1
                and (product_dates[1:] == product_dates[:-1]).any()
            ):
                raise ValueError(
                    f"{product}: duplicate dates in the dataframe passed to backtest() — "
                    "portfolio backtesting requires one row per (date, product)"
                )
            t_idx = np.searchsorted(union_dates, product_dates)
            t_idxs.append(t_idx)
            price_mat[t_idx, j] = close_arr[pos]
            # scatter every registered range of this product onto its rows;
            # rows covered by no range keep weight nan (no instruction → drift)
            for key_range, reg in entries_by_product[product]:
                # min. safe guard to avoid attaching weights to the wrong rows:
                # every registered date must be present in this product's rows
                if not np.isin(reg["dates"], product_dates).all():
                    raise ValueError(
                        f"{product}: registered dates from data_range {key_range} are "
                        "missing from the dataframe passed to backtest()"
                    )
                # rows for the current time_segment/period
                seg_rows = np.searchsorted(product_dates, reg["dates"])
                weight_mat[t_idx[seg_rows], j] = reg["weight"]
                # mark every registered row as in-universe — INCLUDING rows
                # whose weight is nan ("hold"); that is exactly what separates
                # them from uncovered (out-of-universe) rows
                in_universe[t_idx[seg_rows], j] = True

        # ================================================================
        # Ragged panel check: nan closes are only allowed as a PREFIX
        # (listed mid-period) and/or a SUFFIX (delisted/halted mid-period
        # — the kernel force-closes any position left there); a nan
        # BETWEEN a product's first and last bar is a mid-series gap the
        # kernel cannot handle (no price to fill orders / mark equity)
        # ================================================================
        for j, product in enumerate(products):
            non_nan = ~np.isnan(price_mat[:, j])
            first = int(non_nan.argmax())
            last = T - 1 - int(non_nan[::-1].argmax())
            life = non_nan[first : last + 1]
            if not life.all():
                gap_t = first + int((~life).argmax())
                raise ValueError(
                    f"{product}: missing/null close on {union_dates[gap_t]} — portfolio "
                    "backtesting allows missing bars only BEFORE a product's first bar "
                    "(listed mid-period) or AFTER its last bar (delisted/halted); "
                    "mid-series gaps are not supported, filter such products out first"
                )

        # ================================================================
        # A weight needs a price to be sized and filled: a weight on a row
        # whose close is null (allowed above as prefix/suffix missingness)
        # would be silently dropped by the kernel — reject it instead
        # ================================================================
        dropped = ~np.isnan(weight_mat) & np.isnan(price_mat)
        if dropped.any():
            t_bad, j_bad = np.argwhere(dropped)[0]
            raise ValueError(
                f"{products[j_bad]}: weight registered on {union_dates[t_bad]} but "
                "its close is missing/null — a weight needs a price to be sized "
                "and filled; drop such rows or register the weight on a priced bar"
            )

        # ================================================================
        # check: positive weights are target fractions of total equity, so a
        # long sum above 100% on a date implies leverage (each segment is a
        # rebalance to new targets, not a spend of leftover free capital) —
        # flag it (typo or deliberate), never alter results
        # ================================================================
        if check:
            positive_weight_sums = np.where(weight_mat > 0.0, weight_mat, 0.0).sum(
                axis=1
            )
            # small tolerance: exactly-100% allocations must not false-positive
            over = positive_weight_sums > 1.0 + 1e-9
            if over.any():
                over_dates = union_dates[over]
                cprint(
                    f"WARNING: positive weights sum above 100% on {over.sum()} date(s) "
                    f"(first: {over_dates[0]}, max: {positive_weight_sums.max():.1%}) — "
                    "weights are target fractions of total equity, so a long "
                    "sum above 100% means leverage.\n"
                    "Pass check=False to backtest() if leverage is intentional.",
                    style=TextStyle.BOLD + RichColor.YELLOW,
                )

        # ================================================================
        # Run the numba kernel once over the whole panel
        # ================================================================
        (
            order_price_mat,
            order_size_mat,
            trade_price_mat,
            trade_size_mat,
            position_mat,
            avg_price_mat,
            cash_arr,
            equity_arr,
        ) = portfolio_backtest_loop_kernel(
            price_mat,
            weight_mat,
            in_universe,
            float(initial_capital),
            compound,
            T,
            P,
        )

        # ================================================================
        # Scatter (T, P) outputs back to the long df's rows
        # ================================================================
        # Outputs: nan = row not backtested (its product was never configured)
        weight_out = np.full(n, np.nan)
        order_price_out = np.full(n, np.nan)
        order_size_out = np.full(n, np.nan)
        trade_price_out = np.full(n, np.nan)
        trade_size_out = np.full(n, np.nan)
        position_out = np.full(n, np.nan)
        avg_price_out = np.full(n, np.nan)
        cash_out = np.full(n, np.nan)
        equity_out = np.full(n, np.nan)

        for j in range(P):
            pos = product_positions[j]
            t_idx = t_idxs[j]
            weight_out[pos] = weight_mat[t_idx, j]
            order_price_out[pos] = order_price_mat[t_idx, j]
            order_size_out[pos] = order_size_mat[t_idx, j]
            trade_price_out[pos] = trade_price_mat[t_idx, j]
            trade_size_out[pos] = trade_size_mat[t_idx, j]
            position_out[pos] = position_mat[t_idx, j]
            avg_price_out[pos] = avg_price_mat[t_idx, j]
            # portfolio-level values, broadcast to every configured row of the date
            cash_out[pos] = cash_arr[t_idx]
            equity_out[pos] = equity_arr[t_idx]

        # ================================================================
        # Assign output arrays back to dataframe columns
        # ================================================================
        def _make_series(name: str, values: np.ndarray) -> nw.Series:
            return nw.new_series(name, values, nw.Float64, backend=native_backend)

        result_df = df.with_columns(
            _make_series("weight", weight_out),
            _make_series("order_price", order_price_out),
            _make_series("order_size", order_size_out),
            _make_series("trade_price", trade_price_out),
            _make_series("trade_size", trade_size_out),
            _make_series("position", position_out),
            _make_series("avg_price", avg_price_out),
            _make_series("cash", cash_out),
            _make_series("equity", equity_out),
        )
        return result_df.to_native()
