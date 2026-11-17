"""Tests PortfolioBacktestMixin.backtest(): the step that pivots the registered
weights into the kernel's (date, product) matrices and scatters its outputs back."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

    from pytest_mock import MockerFixture

import datetime
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest


def _load_mixin_module() -> ModuleType:
    name = "pfund.backtest.portfolio_backtest_mixin"
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    # TODO: drop this fallback once `pfund.components` is importable again. Its
    # `__init__` currently fails during the refactor, so the one module the
    # mixin needs from it (bar_dataframe, for PIVOT_COLS) is loaded straight
    # from its file for the duration of the mixin's import.
    dep_name = "pfund.components.bar_dataframe"
    dep_path = (
        Path(__file__).parents[3] / "src" / "pfund" / "components" / "bar_dataframe.py"
    )
    spec = importlib.util.spec_from_file_location(dep_name, dep_path)
    assert spec is not None and spec.loader is not None
    dep = importlib.util.module_from_spec(spec)
    sys.modules[dep_name] = dep
    try:
        spec.loader.exec_module(dep)
        return importlib.import_module(name)
    finally:
        sys.modules.pop(dep_name, None)


m = _load_mixin_module()
Mixin = m.PortfolioBacktestMixin

NAN = np.nan
N = 6
CAPITAL = 1000.0
BASE_PRICE = {"BTC": 100.0, "ETH": 10.0, "SOL": 50.0}
OUTPUT_COLS = [
    "weight",
    "order_price",
    "order_size",
    "trade_price",
    "trade_size",
    "position",
    "avg_price",
    "cash",
    "equity",
]

type Weights = list[float | None]


class KernelInputs(NamedTuple):
    """portfolio_backtest_loop_kernel's positional arguments, in order."""

    price: Any  # [T, P]
    weight: Any  # [T, P]
    in_universe: Any  # [T, P]
    initial_capital: Any
    compound: Any
    T: int
    P: int


def day(i: int) -> datetime.date:
    return datetime.date(2024, 1, 1) + datetime.timedelta(days=i)


def rows(first: int, last: int) -> tuple[datetime.date, datetime.date]:
    """data_range covering row indices [first, last] of a make_df() product."""
    return (day(first), day(last))


def make_df(
    products: tuple[str, ...] = ("BTC",),
    *,
    closes: dict[str, list[float | None]] | None = None,
    n: int = N,
) -> pl.DataFrame:
    """Long df sorted by date with the products interleaved on every date.

    A product's close is flat at its BASE_PRICE unless `closes` spells it out
    per row (None = a null close).
    """
    closes = closes or {}
    records: list[dict[str, Any]] = []
    for i in range(n):
        for product in products:
            close = closes[product][i] if product in closes else BASE_PRICE[product]
            records.append(
                {
                    "date": datetime.datetime.combine(day(i), datetime.time()),
                    "product": product,
                    "resolution": "1d",
                    "close": close,
                }
            )
    return pl.DataFrame(records, schema_overrides={"close": pl.Float64})


def drop_rows(df: pl.DataFrame, product: str, first: int, last: int) -> pl.DataFrame:
    """Remove a product's rows [first, last], e.g. to list it late or delist it."""
    in_range = pl.col("date").dt.date().is_between(day(first), day(last))
    return df.filter(~((pl.col("product") == product) & in_range))


def configure(
    df: pl.DataFrame,
    product: str = "BTC",
    *,
    weight: Weights | None = None,
    data_range: tuple[datetime.date, datetime.date] | None = None,
) -> None:
    """Register one segment of one product the way a user would.

    `weight` is positional over ALL of the product's rows in `df`;
    create_weight slices it to data_range. None = nan on every row (hold).
    """
    sub = df.filter(pl.col("product") == product)
    values = [None] * len(sub) if weight is None else weight
    Mixin.create_weight(
        sub, weight=pl.Series("weight", values, dtype=pl.Float64), data_range=data_range
    )


def backtest(df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    return Mixin.backtest(df, **{"initial_capital": CAPITAL, **kwargs})


def col(result: pl.DataFrame, name: str, product: str = "BTC") -> Any:
    return result.filter(pl.col("product") == product).get_column(name).to_numpy()


def eq(actual: Any, expected: Any) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        equal_nan=True,
    )


def eq_bool(actual: Any, expected: str) -> None:
    """Compare a bool array with a 'TTFF'-style string."""
    assert "".join("T" if v else "F" for v in actual) == expected


@pytest.fixture(autouse=True)
def fresh_registry() -> Iterator[None]:
    m._clear_registry()
    yield
    m._clear_registry()


@pytest.fixture
def kernel_inputs(mocker: MockerFixture) -> Callable[[], list[KernelInputs]]:
    """Record what backtest() feeds the kernel; the real kernel still runs."""
    mock = mocker.patch.object(
        m,
        "portfolio_backtest_loop_kernel",
        side_effect=m.portfolio_backtest_loop_kernel,
    )
    return lambda: [KernelInputs(*call.args) for call in mock.call_args_list]


class TestErrors:
    def test_nothing_registered(self):
        with pytest.raises(ValueError, match="backtest\\(\\) cannot run"):
            backtest(make_df())

    @pytest.mark.parametrize("capital", [0, -1, -0.5])
    def test_non_positive_initial_capital(self, capital: float):
        df = make_df()
        configure(df)
        with pytest.raises(ValueError, match="'initial_capital' must be positive"):
            backtest(df, initial_capital=capital)

    def test_more_than_one_resolution(self):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC")
        mixed = df.with_columns(
            pl.when(pl.col("product") == "ETH")
            .then(pl.lit("1h"))
            .otherwise(pl.col("resolution"))
            .alias("resolution")
        )
        with pytest.raises(ValueError, match="requires a single resolution"):
            backtest(mixed)

    def test_configured_product_absent_from_the_df(self):
        configure(make_df(("BTC", "ETH")), "BTC")
        with pytest.raises(
            ValueError, match="has no rows in the dataframe passed to backtest"
        ):
            backtest(make_df(("ETH",)))

    def test_registered_dates_missing_from_the_df(self):
        df = make_df()
        configure(df)
        with pytest.raises(
            ValueError, match="are missing from the dataframe passed to backtest"
        ):
            backtest(drop_rows(df, "BTC", 5, 5))

    def test_duplicate_dates_within_a_product(self):
        df = make_df()
        configure(df)
        doubled = pl.concat([df, df]).sort("date")
        with pytest.raises(
            ValueError, match="duplicate dates in the dataframe passed to backtest"
        ):
            backtest(doubled)

    def test_missing_row_in_the_middle_of_a_products_series(self):
        df = drop_rows(make_df(("BTC", "ETH")), "ETH", 2, 2)
        configure(df, "BTC")
        configure(df, "ETH")
        with pytest.raises(
            ValueError, match=r"ETH: missing/null close on .*mid-series"
        ):
            backtest(df)

    def test_null_close_in_the_middle_of_a_products_series(self):
        df = make_df(closes={"BTC": [100, 100, None, 100, 100, 100]})
        configure(df)
        with pytest.raises(
            ValueError, match=r"BTC: missing/null close on .*mid-series"
        ):
            backtest(df)

    def test_weight_on_a_null_close(self):
        df = make_df(closes={"BTC": [None, 100, 100, 100, 100, 100]})
        configure(df, weight=[0.5, None, None, None, None, None])
        with pytest.raises(
            ValueError, match=r"BTC: weight registered on .*missing/null"
        ):
            backtest(df)

    def test_kernel_error_reaches_the_user(self):
        # BTC is kept (nan) while holding all the capital: ETH cannot be sized
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", weight=[1.0, None, None, None, None, None])
        configure(df, "ETH", weight=[None, 0.5, None, None, None, None])
        with pytest.raises(ValueError, match="free capital"):
            backtest(df)


class TestOutputFrame:
    def test_appends_the_output_columns_and_keeps_the_rows(self):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", weight=[0.5, None, None, None, None, None])
        result = backtest(df)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == [*df.columns, *OUTPUT_COLS]
        assert result.select(df.columns).equals(df)
        assert all(result.schema[c] == pl.Float64 for c in OUTPUT_COLS)

    def test_input_df_is_not_modified(self):
        df = make_df()
        before = df.clone()
        configure(df, weight=[0.5, None, None, None, None, None])
        backtest(df)
        assert df.equals(before)

    def test_weight_column_echoes_the_registered_weight(self):
        df = make_df()
        configure(df, weight=[0.5, None, 0.0, None, -0.2, None])
        eq(col(backtest(df), "weight"), [0.5, NAN, 0.0, NAN, -0.2, NAN])

    def test_unconfigured_product_is_all_nan(self):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", weight=[0.5, None, None, None, None, None])
        result = backtest(df)
        for name in OUTPUT_COLS:
            eq(col(result, name, "ETH"), [NAN] * N)
        # a configured product reports a real (flat) position instead of nan
        eq(col(result, "position", "BTC"), [0, 5, 5, 5, 5, 5])

    def test_each_product_gets_its_own_results_on_its_own_rows(self):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", weight=[0.5, None, None, None, None, None])
        configure(df, "ETH", weight=[None, -0.2, None, None, None, None])
        result = backtest(df)
        eq(col(result, "position", "BTC"), [0, 5, 5, 5, 5, 5])
        eq(col(result, "trade_price", "BTC"), [NAN, 100, NAN, NAN, NAN, NAN])
        # ETH is sized on the capital BTC (kept from day 1) leaves free: 500
        eq(col(result, "position", "ETH"), [0, 0, -10, -10, -10, -10])
        eq(col(result, "trade_price", "ETH"), [NAN, NAN, 10, NAN, NAN, NAN])

    def test_cash_and_equity_are_shared_by_every_product_of_a_date(self):
        df = make_df(("BTC", "ETH"), closes={"BTC": [100, 100, 120, 120, 90, 90]})
        configure(df, "BTC", weight=[0.5, None, None, None, None, None])
        configure(df, "ETH", weight=[0.3, None, None, None, None, None])
        result = backtest(df)
        eq(col(result, "cash", "BTC"), [1000, 200, 200, 200, 200, 200])
        eq(col(result, "equity", "BTC"), [1000, 1000, 1100, 1100, 950, 950])
        for name in ("cash", "equity"):
            eq(col(result, name, "ETH"), col(result, name, "BTC"))

    def test_rows_grouped_by_product_instead_of_interleaved(self):
        # backtest() must not assume any particular row layout across products
        df = make_df(("BTC", "ETH")).sort("date", "product", descending=[False, True])
        configure(df, "BTC", weight=[0.5, None, None, None, None, None])
        configure(df, "ETH", weight=[0.2, None, None, None, None, None])
        result = backtest(df)
        eq(col(result, "position", "BTC"), [0, 5, 5, 5, 5, 5])
        eq(col(result, "position", "ETH"), [0, 20, 20, 20, 20, 20])

    def test_registration_order_does_not_matter(self):
        df = make_df(("BTC", "ETH"), closes={"BTC": [100, 110, 90, 120, 100, 100]})
        btc: Weights = [0.5, None, 0.3, None, None, None]
        eth: Weights = [0.2, None, 0.6, None, None, None]
        configure(df, "BTC", weight=btc)
        configure(df, "ETH", weight=eth)
        forward = backtest(df)

        m._clear_registry()
        configure(df, "ETH", weight=eth)
        configure(df, "BTC", weight=btc)
        backward = backtest(df)
        for product in ("BTC", "ETH"):
            for name in OUTPUT_COLS:
                eq(col(forward, name, product), col(backward, name, product))

    def test_registration_covering_no_rows_leaves_the_product_flat(self):
        df = make_df()
        configure(
            df, data_range=(datetime.date(2030, 1, 1), datetime.date(2030, 1, 31))
        )
        result = backtest(df)
        eq(col(result, "position"), [0] * N)
        eq(col(result, "order_size"), [NAN] * N)
        eq(col(result, "equity"), [CAPITAL] * N)

    def test_pandas_backend_with_a_non_default_index(self):
        pl_df = make_df(("BTC", "ETH"))
        configure(pl_df, "BTC", weight=[0.5, None, None, 0.0, None, None])
        expected = backtest(pl_df)

        m._clear_registry()
        pd_df = pl_df.to_pandas()
        pd_df.index = pd_df.index[::-1]  # labels no longer match positions
        sub = pd_df[pd_df["product"] == "BTC"]
        weight = pd.Series([0.5, NAN, NAN, 0.0, NAN, NAN], index=sub.index)
        Mixin.create_weight(sub, weight=weight)
        result = Mixin.backtest(pd_df, initial_capital=CAPITAL)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == expected.columns
        assert list(result.index) == list(pd_df.index)
        for name in OUTPUT_COLS:
            eq(result[name].to_numpy(), expected.get_column(name).to_numpy())


class TestKernelInputs:
    def test_one_call_over_the_whole_panel(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC")
        configure(df, "ETH")
        backtest(df, initial_capital=5000, compound=False)
        (call,) = kernel_inputs()
        assert (call.T, call.P) == (N, 2)
        assert call.initial_capital == 5000.0
        assert isinstance(call.initial_capital, float)
        assert call.compound is False

    def test_defaults(self, kernel_inputs: Callable[[], list[KernelInputs]]):
        df = make_df()
        configure(df)
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        assert call.initial_capital == 1_000_000.0
        assert call.compound is True

    def test_every_matrix_is_date_by_product_with_the_right_dtype(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC", "ETH", "SOL"))
        configure(df, "BTC", data_range=rows(1, 3))
        configure(df, "SOL")
        backtest(df)
        (call,) = kernel_inputs()
        for name, dtype in (
            ("price", np.float64),
            ("weight", np.float64),
            ("in_universe", np.bool_),
        ):
            value = getattr(call, name)
            assert value.shape == (N, 2), name
            assert value.dtype == dtype, name

    def test_columns_are_products_in_first_registration_order(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC", "ETH", "SOL"))
        configure(df, "SOL")
        configure(df, "BTC", data_range=rows(0, 2))
        configure(df, "ETH")
        configure(df, "BTC", data_range=rows(3, 5))
        backtest(df)
        (call,) = kernel_inputs()
        eq(call.price[0], [50, 100, 10])

    def test_unconfigured_product_is_not_in_the_panel(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC", "ETH"))
        configure(df, "ETH")
        backtest(df)
        (call,) = kernel_inputs()
        assert call.P == 1
        eq(call.price[:, 0], [10] * N)

    def test_dates_come_only_from_configured_products(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        # ETH trades on two more dates than BTC, but it is not in the portfolio
        df = drop_rows(make_df(("BTC", "ETH")), "BTC", 4, 5)
        configure(df, "BTC")
        backtest(df)
        (call,) = kernel_inputs()
        assert call.T == 4

    def test_prices_are_each_products_closes_by_date(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(
            ("BTC", "ETH"),
            closes={
                "BTC": [100, 101, 102, 103, 104, 105],
                "ETH": [10, 11, 12, 13, 14, 15],
            },
        )
        configure(df, "BTC")
        configure(df, "ETH")
        backtest(df)
        (call,) = kernel_inputs()
        eq(call.price[:, 0], [100, 101, 102, 103, 104, 105])
        eq(call.price[:, 1], [10, 11, 12, 13, 14, 15])

    def test_weights_land_on_their_own_dates(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", weight=[0.1, None, 0.3, None, None, 0.0])
        configure(df, "ETH", weight=[None, -0.2, None, None, 0.4, None])
        backtest(df)
        (call,) = kernel_inputs()
        eq(call.weight[:, 0], [0.1, NAN, 0.3, NAN, NAN, 0.0])
        eq(call.weight[:, 1], [NAN, -0.2, NAN, NAN, 0.4, NAN])

    def test_covered_rows_are_in_universe_even_with_a_nan_weight(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        configure(df)
        backtest(df)
        (call,) = kernel_inputs()
        eq(call.weight[:, 0], [NAN] * N)
        eq_bool(call.in_universe[:, 0], "TTTTTT")

    def test_product_delisted_mid_period_has_no_price_and_no_universe(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = drop_rows(make_df(("BTC", "ETH")), "ETH", 4, 5)
        configure(df, "BTC")
        configure(df, "ETH")
        backtest(df)
        (call,) = kernel_inputs()
        assert call.T == N
        eq(call.price[:, 1], [10, 10, 10, 10, NAN, NAN])
        eq_bool(call.in_universe[:, 1], "TTTTFF")
        eq_bool(call.in_universe[:, 0], "TTTTTT")

    def test_product_listed_mid_period_has_no_price_and_no_universe(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = drop_rows(make_df(("BTC", "ETH")), "ETH", 0, 1)
        configure(df, "BTC")
        configure(df, "ETH", weight=[0.5, None, None, None])
        backtest(df)
        (call,) = kernel_inputs()
        eq(call.price[:, 1], [NAN, NAN, 10, 10, 10, 10])
        eq(call.weight[:, 1], [NAN, NAN, 0.5, NAN, NAN, NAN])
        eq_bool(call.in_universe[:, 1], "FFTTTT")

    def test_registered_row_with_a_null_close_is_out_of_universe(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        # the kernel needs a price wherever a product is in the universe
        df = make_df(closes={"BTC": [None, 100, 100, 100, None, None]})
        configure(df)
        backtest(df)
        (call,) = kernel_inputs()
        eq(call.price[:, 0], [NAN, 100, 100, 100, NAN, NAN])
        eq_bool(call.in_universe[:, 0], "FTTTFF")


class TestSegments:
    def test_rows_outside_the_segment_are_out_of_universe(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        configure(df, weight=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], data_range=rows(2, 3))
        backtest(df)
        (call,) = kernel_inputs()
        # the weight series is positional over the df, then sliced to the range
        eq(call.weight[:, 0], [NAN, NAN, 0.3, 0.4, NAN, NAN])
        eq_bool(call.in_universe[:, 0], "FFTTFF")

    def test_contiguous_segments_cover_every_row(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        weight: Weights = [0.1, None, 0.3, 0.4, None, 0.6]
        configure(df, weight=weight, data_range=rows(0, 2))
        configure(df, weight=weight, data_range=rows(3, 5))
        backtest(df)
        (call,) = kernel_inputs()
        eq(call.weight[:, 0], [0.1, NAN, 0.3, 0.4, NAN, 0.6])
        eq_bool(call.in_universe[:, 0], "TTTTTT")

    def test_gap_rows_between_segments_are_out_of_universe(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        weight: Weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        configure(df, weight=weight, data_range=rows(0, 1))
        configure(df, weight=weight, data_range=rows(4, 5))
        backtest(df)
        (call,) = kernel_inputs()
        eq(call.weight[:, 0], [0.1, 0.2, NAN, NAN, 0.5, 0.6])
        eq_bool(call.in_universe[:, 0], "TTFFTT")

    def test_each_product_has_its_own_universe(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", data_range=rows(0, 2))
        configure(df, "ETH", data_range=rows(2, 5))
        backtest(df)
        (call,) = kernel_inputs()
        eq_bool(call.in_universe[:, 0], "TTTFFF")
        eq_bool(call.in_universe[:, 1], "FFTTTT")

    def test_segment_registration_order_does_not_matter(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        weight: Weights = [0.5, None, None, 0.2, None, None]
        configure(df, weight=weight, data_range=rows(0, 2))
        configure(df, weight=weight, data_range=rows(3, 5))
        forward = backtest(df)

        m._clear_registry()
        configure(df, weight=weight, data_range=rows(3, 5))
        configure(df, weight=weight, data_range=rows(0, 2))
        backward = backtest(df)

        first, second = kernel_inputs()
        for name, value in first._asdict().items():
            np.testing.assert_array_equal(value, getattr(second, name), err_msg=name)
        assert forward.equals(backward, null_equal=True)

    def test_empty_segment_is_skipped(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        configure(df, weight=[0.5, None, None, None, None, None], data_range=rows(0, 5))
        configure(
            df, data_range=(datetime.date(2030, 1, 1), datetime.date(2030, 1, 31))
        )
        result = backtest(df)
        (call,) = kernel_inputs()
        eq_bool(call.in_universe[:, 0], "TTTTTT")
        eq(col(result, "position"), [0, 5, 5, 5, 5, 5])

    def test_segment_registered_on_an_expanding_window(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        # the dynamic-period pattern: each period is configured on the history
        # known so far, then ONE backtest() runs over the full df
        df = make_df(("BTC", "ETH"))
        weight: Weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for first, last in ((0, 1), (2, 3), (4, 5)):
            window = df.filter(pl.col("date").dt.date() <= day(last))
            configure(
                window, "BTC", weight=weight[: last + 1], data_range=rows(first, last)
            )
        backtest(df)
        (call,) = kernel_inputs()
        assert call.P == 1
        eq(call.weight[:, 0], weight)
        eq_bool(call.in_universe[:, 0], "TTTTTT")


class TestLeverageCheck:
    """check=True warns when a date's positive weights sum above 100%."""

    @staticmethod
    def _configure(btc: float, eth: float) -> pl.DataFrame:
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", weight=[btc, None, None, None, None, None])
        configure(df, "ETH", weight=[eth, None, None, None, None, None])
        return df

    def test_warns_above_100_percent(self, mocker: MockerFixture):
        cprint = mocker.patch.object(m, "cprint")
        backtest(self._configure(0.6, 0.6))
        cprint.assert_called_once()
        message = cprint.call_args.args[0]
        assert "positive weights sum above 100% on 1 date(s)" in message
        assert "max: 120.0%" in message

    def test_exactly_100_percent_does_not_warn(self, mocker: MockerFixture):
        cprint = mocker.patch.object(m, "cprint")
        backtest(self._configure(0.7, 0.3))
        cprint.assert_not_called()

    def test_shorts_do_not_count(self, mocker: MockerFixture):
        cprint = mocker.patch.object(m, "cprint")
        backtest(self._configure(0.9, -0.9))
        cprint.assert_not_called()

    def test_check_false_is_silent(self, mocker: MockerFixture):
        cprint = mocker.patch.object(m, "cprint")
        backtest(self._configure(0.6, 0.6), check=False)
        cprint.assert_not_called()

    def test_warning_never_changes_the_result(self, mocker: MockerFixture):
        mocker.patch.object(m, "cprint")
        df = self._configure(0.6, 0.6)
        assert backtest(df).equals(backtest(df, check=False), null_equal=True)


class TestEndToEnd:
    """backtest() + the real kernel, on the behaviours that depend on both."""

    def test_weight_is_filled_on_the_next_date_at_the_placing_close(self):
        df = make_df(closes={"BTC": [100, 110, 120, 120, 120, 120]})
        configure(df, weight=[0.5, None, None, None, None, None])
        result = backtest(df)
        eq(col(result, "order_price"), [100, NAN, NAN, NAN, NAN, NAN])
        eq(col(result, "order_size"), [5, NAN, NAN, NAN, NAN, NAN])
        eq(col(result, "trade_price"), [NAN, 100, NAN, NAN, NAN, NAN])
        eq(col(result, "trade_size"), [NAN, 5, NAN, NAN, NAN, NAN])
        eq(col(result, "position"), [0, 5, 5, 5, 5, 5])
        eq(col(result, "avg_price"), [NAN, 100, 100, 100, 100, 100])
        eq(col(result, "cash"), [1000, 500, 500, 500, 500, 500])
        eq(col(result, "equity"), [1000, 1050, 1100, 1100, 1100, 1100])

    def test_compound_flag_changes_the_sizing_capital(self):
        df = make_df(closes={"BTC": [100, 200, 200, 200, 200, 200]})
        configure(df, weight=[1.0, 1.0, None, None, None, None])
        # 100% of the doubled equity is still 10 units
        eq(col(backtest(df), "position"), [0, 10, 10, 10, 10, 10])
        # 100% of the initial 1000 at 200 is 5 units
        eq(col(backtest(df, compound=False), "position"), [0, 10, 5, 5, 5, 5])

    def test_position_carries_across_contiguous_segments(self):
        df = make_df()
        weight: Weights = [0.5, None, None, None, None, None]
        configure(df, weight=weight, data_range=rows(0, 2))
        configure(df, weight=weight, data_range=rows(3, 5))
        result = backtest(df)
        eq(col(result, "position"), [0, 5, 5, 5, 5, 5])
        eq(col(result, "trade_size"), [NAN, 5, NAN, NAN, NAN, NAN])

    def test_position_is_closed_at_the_close_of_the_first_gap_row(self):
        # the close order is placed on the gap row and filled the date after
        df = make_df(closes={"BTC": [100, 100, 110, 120, 120, 120]})
        weight: Weights = [0.5, None, None, None, None, None]
        configure(df, weight=weight, data_range=rows(0, 1))
        configure(df, weight=weight, data_range=rows(4, 5))
        result = backtest(df)
        eq(col(result, "order_size"), [5, NAN, -5, NAN, NAN, NAN])
        eq(col(result, "trade_price"), [NAN, 100, NAN, 110, NAN, NAN])
        eq(col(result, "trade_size"), [NAN, 5, NAN, -5, NAN, NAN])
        eq(col(result, "position"), [0, 5, 5, 0, 0, 0])
        eq(col(result, "cash"), [1000, 500, 500, 1050, 1050, 1050])

    def test_position_is_closed_when_the_last_segment_ends_early(self):
        df = make_df()
        configure(df, weight=[0.5, None, None, None, None, None], data_range=rows(0, 2))
        result = backtest(df)
        eq(col(result, "order_size"), [5, NAN, NAN, -5, NAN, NAN])
        eq(col(result, "position"), [0, 5, 5, 5, 0, 0])

    def test_kept_position_is_set_aside_when_sizing_the_others(self):
        # BTC is kept (covered, nan) while worth 1000 of the 1500 pot: ETH's
        # 100% is 100% of the other 500
        df = make_df(("BTC", "ETH"), closes={"BTC": [100, 200, 200, 200, 200, 200]})
        configure(df, "BTC", weight=[0.5, None, None, None, None, None])
        configure(df, "ETH", weight=[None, 1.0, None, None, None, None])
        result = backtest(df)
        eq(col(result, "equity"), [1000, 1500, 1500, 1500, 1500, 1500])
        eq(col(result, "position", "ETH"), [0, 0, 50, 50, 50, 50])
        eq(col(result, "cash"), [1000, 500, 0, 0, 0, 0])

    def test_delisting_without_rows_shows_only_in_cash_and_equity(self):
        # ETH has no row after day 2, so its write-off has nowhere to be shown
        df = drop_rows(make_df(("BTC", "ETH")), "ETH", 3, 5)
        configure(df, "BTC", weight=[0.5, None, None, None, None, None])
        configure(df, "ETH", weight=[0.5, None, None])
        result = backtest(df)
        eq(col(result, "position", "ETH"), [0, 50, 50])
        eq(col(result, "order_size", "ETH"), [50, NAN, NAN])
        eq(col(result, "trade_size", "ETH"), [NAN, 50, NAN])
        eq(col(result, "equity", "BTC"), [1000, 1000, 1000, 500, 500, 500])
        eq(col(result, "cash", "BTC"), [1000, 0, 0, 0, 0, 0])
        eq(col(result, "position", "BTC"), [0, 5, 5, 5, 5, 5])

    def test_delisting_with_null_close_rows_shows_the_write_off(self):
        df = make_df(("BTC", "ETH"), closes={"ETH": [10, 10, 10, None, None, None]})
        configure(df, "BTC", weight=[0.5, None, None, None, None, None])
        configure(df, "ETH", weight=[0.5, None, None, None, None, None])
        result = backtest(df)
        eq(col(result, "order_price", "ETH"), [10, NAN, NAN, 0, NAN, NAN])
        eq(col(result, "order_size", "ETH"), [50, NAN, NAN, -50, NAN, NAN])
        eq(col(result, "trade_price", "ETH"), [NAN, 10, NAN, NAN, 0, NAN])
        eq(col(result, "trade_size", "ETH"), [NAN, 50, NAN, NAN, -50, NAN])
        eq(col(result, "position", "ETH"), [0, 50, 50, 50, 0, 0])
        eq(col(result, "equity", "ETH"), [1000, 1000, 1000, 500, 500, 500])

    def test_null_close_rows_do_not_block_the_other_products(self):
        # a kept position whose close goes null used to turn every order of
        # that date into nan; ETH must still be sized and filled
        df = make_df(("BTC", "ETH"), closes={"BTC": [100, 100, None, None, None, None]})
        configure(df, "BTC", weight=[0.5, None, None, None, None, None])
        configure(df, "ETH", weight=[None, None, 0.5, None, None, None])
        result = backtest(df)
        eq(col(result, "order_size", "ETH"), [NAN, NAN, 25, NAN, NAN, NAN])
        eq(col(result, "position", "ETH"), [0, 0, 0, 25, 25, 25])
        eq(col(result, "position", "BTC"), [0, 5, 5, 0, 0, 0])
        eq(col(result, "equity", "ETH"), [1000, 1000, 500, 500, 500, 500])
