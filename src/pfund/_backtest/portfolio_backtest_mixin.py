# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportReturnType=false, reportMissingTypeArgument=false, reportImplicitStringConcatenation=false, reportUnknownParameterType=false, reportArgumentType=false, reportCallIssue=false
import datetime

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT, IntoSeries
from pfund_kit.style import RichColor, TextStyle, cprint

from pfund._backtest.portfolio_backtest_kernel import portfolio_backtest_loop_kernel
from pfund._backtest.product_backtest_mixin import (
    _data_range_mask,
    _group_positions,
    _series_to_positional_float64,
    _to_float64,
    _validate_data_range,
)

# one entry per create_weight() call
_registry: dict[tuple, dict] = {}
# ((start_date, end_date), "BYBIT_BTC_USDT_PERPETUAL"): {  # key = (data_range, product); range resolved
#   "data_range": (start, end) | None,  # as passed to create_weight();
#                 None = full-span single-shot (strict full-row guard in backtest())
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
        weight: IntoSeries | float,
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
            weight:
                scalar → one rebalance instruction at the LAST row in
                data_range; the earlier rows in range get nan (drift).
                native series (pandas or polars) matching the dataframe's
                backend → positional weights over the rows in data_range,
                length must match the row count exactly.
            data_range: (start_date, end_date), both inclusive — the rows this
                call covers. A range selecting no rows (e.g. a delisted
                product still in the universe) is a no-op.
                None (default) → single-shot configuration over ALL rows of
                this df: wipes the product's previously registered ranges
                (reconfigure + rerun).
        """
        df = nw.from_native(self)
        if "date" not in df.columns:
            raise ValueError(
                "portfolio backtesting requires a 'date' column — it pins the "
                "registered rows so backtest() can verify row alignment"
            )
        # weights are positional (a scalar lands at the range's LAST row): an
        # unsorted df would silently anchor it to the wrong date
        if not df.get_column("date").is_sorted():
            raise ValueError(
                "the dataframe passed to create_weight() must be sorted by "
                "'date' (ascending)"
            )
        if data_range is not None:
            data_range = _validate_data_range(data_range)
        # an empty df is the "selects no rows" case taken to its limit (e.g. a
        # product from a static universe with no rows in this window) — same
        # no-op as a data_range selecting nothing; no product key can be
        # derived from it, so it must be handled before _get_registry_key
        if len(df) == 0:
            return self
        key = _get_registry_key(df, data_range)
        key_range, product = key
        date_arr = df.get_column("date").to_numpy()

        if data_range is None:
            target_pos = np.arange(len(df))
        else:
            start, end = data_range
            target_pos = np.flatnonzero(_data_range_mask(date_arr, start, end))
        if len(target_pos) == 0:
            # nothing to register, e.g. a delisted product still in the
            # universe has no rows in the current period — no-op
            return self
        target_dates = date_arr[target_pos]
        # duplicate dates (e.g. one product at multiple resolutions) would
        # double-write the same panel row at backtest() — last value silently
        # wins; reject at registration instead
        if len(np.unique(target_dates)) != len(target_dates):
            raise ValueError(
                f"{product}: duplicate dates in the registered rows — the df likely "
                "holds multiple resolutions; portfolio backtesting requires a "
                "single resolution (one row per (date, product))"
            )

        if isinstance(weight, (int, float)) and not isinstance(weight, bool):
            # one rebalance instruction at the range's last row, drift before it
            weight_arr = np.full(len(target_pos), np.nan)
            weight_arr[-1] = float(weight)
        else:
            weight_arr = _series_to_positional_float64(
                weight, len(target_pos), "weight"
            )
        if np.isinf(weight_arr).any():
            raise ValueError(
                "'weight' must not contain inf/-inf — weights are fractions of "
                "the portfolio's sizing capital (nan = hold, 0.0 = close)"
            )

        if data_range is None:
            # legacy single-shot configuration: wipe the product's previously
            # registered ranges (reconfigure + rerun) and cover all rows
            for registered_key in [k for k in _registry if k[1] == product]:
                del _registry[registered_key]
        else:
            # a row's weight is registered once — the past is never rewritten
            for (registered_range, registered_product), entry in _registry.items():
                if registered_product != product:
                    continue
                if np.isin(target_dates, entry["dates"]).any():
                    raise ValueError(
                        f"{product}: data_range {key_range} overlaps the already-registered "
                        f"{registered_range} — a row's weight can only be registered once"
                    )

        _registry[key] = {
            # data_range as passed: None = full-span single-shot configuration
            "data_range": data_range,
            "weight": weight_arr,
            "dates": target_dates,
        }

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

        # ================================================================
        # Data arrays from the df backtest() is called on, converted once
        # ================================================================
        if "date" not in df.columns:
            raise ValueError(
                "portfolio backtesting requires a 'date' column — it is needed to "
                "verify row alignment against the registered weights"
            )
        # weight alignment maps registered dates to panel rows via
        # searchsorted over each product's dates: an unsorted df would
        # silently scatter weights onto the wrong rows
        if not df.get_column("date").is_sorted():
            raise ValueError(
                "the dataframe passed to backtest() must be sorted by "
                "'date' (ascending)"
            )
        close_arr = _to_float64(df.get_column("close"))
        date_arr = df.get_column("date").to_numpy()

        # ================================================================
        # Single resolution: mixing resolutions is not meaningful for
        # portfolio management (one shared equity pot is marked per date)
        # ================================================================
        resolutions = df.get_column("resolution").unique().to_list()
        if len(resolutions) > 1:
            raise ValueError(
                f"portfolio backtesting requires a single resolution, got {sorted(resolutions)} — "
                "mixing resolutions is not supported; resample your data to one resolution first"
            )

        # Group rows by product once (O(n log n)) instead of an
        # O(products * n) boolean mask rebuilt per product below.
        positions_by_product = _group_positions(
            [df.get_column("product").to_numpy()], n
        )

        # ================================================================
        # Per-product row positions (products in first-registration order)
        # ================================================================
        products: list[str] = []
        for _, product in _registry:
            if product not in products:
                products.append(product)
        P = len(products)
        product_positions: list[np.ndarray] = []
        for product in products:
            # positions of this product's rows (ascending), computed once above
            pos = positions_by_product.get((product,))
            if pos is None:
                raise ValueError(
                    f"{product} has no rows in the dataframe passed to backtest()"
                )
            product_positions.append(pos)

        # ================================================================
        # Pivot long df → date-major (T, P) matrices for the kernel
        # ================================================================
        union_dates = np.unique(
            np.concatenate([date_arr[pos] for pos in product_positions])
        )
        T = len(union_dates)
        close_mat = np.full((T, P), np.nan)
        weight_mat = np.full((T, P), np.nan)
        # each product's row positions in the (T, P) panel; reused to scatter
        # outputs back, so round-trip row alignment is structural
        t_idxs: list[np.ndarray] = []
        for j, product in enumerate(products):
            pos = product_positions[j]
            product_dates = date_arr[pos]
            # duplicate (date, product) rows would collapse onto one panel row
            # (searchsorted maps to the first occurrence, close last-write-wins);
            # the registration guard only sees the df passed to create_weight(),
            # so guard this df too. Dates are ascending → duplicates are adjacent.
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
            close_mat[t_idx, j] = close_arr[pos]
            # scatter every registered range of this product onto its rows;
            # rows covered by no range keep weight nan (no instruction → drift)
            for (key_range, registered_product), entry in _registry.items():
                if registered_product != product:
                    continue
                # min. safe guard to avoid attaching weights to the wrong rows
                if entry["data_range"] is None:
                    # full-span single-shot entry: must pin the product's exact rows
                    if not np.array_equal(product_dates, entry["dates"]):
                        raise ValueError(f"{product}: mismatched dates")
                    seg_rows = np.arange(len(pos))
                else:
                    if not np.isin(entry["dates"], product_dates).all():
                        raise ValueError(
                            f"{product}: registered dates from data_range {key_range} are "
                            "missing from the dataframe passed to backtest()"
                        )
                    # product dates are unique (duplicate-dates guard above)
                    # and ascending, so searchsorted maps the segment's dates
                    # to the product's row indices exactly
                    seg_rows = np.searchsorted(product_dates, entry["dates"])
                weight_mat[t_idx[seg_rows], j] = entry["weight"]

        # ================================================================
        # Ragged panel check: nan closes are only allowed as a PREFIX
        # (listed mid-period) and/or a SUFFIX (delisted/halted mid-period
        # — the kernel force-closes any position left there); a nan
        # BETWEEN a product's first and last bar is a mid-series gap the
        # kernel cannot handle (no price to fill orders / mark equity)
        # ================================================================
        for j, product in enumerate(products):
            non_nan = ~np.isnan(close_mat[:, j])
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
        dropped = ~np.isnan(weight_mat) & np.isnan(close_mat)
        if dropped.any():
            t_bad, j_bad = np.argwhere(dropped)[0]
            raise ValueError(
                f"{products[j_bad]}: weight registered on {union_dates[t_bad]} but "
                "its close is missing/null — a weight needs a price to be sized "
                "and filled; drop such rows or register the weight on a priced bar"
            )

        # ================================================================
        # check: positive weights summing above 100% on a date lever the
        # free capital — flag it (typo or deliberate), never alter results
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
                    "weights are fractions of FREE capital, so this means leverage.\n"
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
            close_mat,
            weight_mat,
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
