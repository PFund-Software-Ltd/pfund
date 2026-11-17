"""Tests ProductBacktestMixin.backtest(): the step that turns the registered
segments into the kernel's per-bar inputs and scatters its outputs back."""

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
    name = "pfund.backtest.product_backtest_mixin"
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
Mixin = m.ProductBacktestMixin

NAN = np.nan
N = 6
BASE_PRICE = {"BTC": 100.0, "ETH": 10.0}
OUTPUT_COLS = [
    "signal",
    "order_price",
    "order_size",
    "trade_price",
    "trade_size",
    "position",
    "avg_price",
    "stop_price",
]


class KernelInputs(NamedTuple):
    """backtest_loop_kernel's positional arguments, in order."""

    open: Any
    high: Any
    low: Any
    close: Any
    volume: Any
    signal: Any
    in_universe: Any
    order_price: Any
    order_quantity: Any
    first_only: Any
    long_only: Any
    short_only: Any
    take_profit: Any
    stop_loss: Any
    time_window: Any
    trailing_stop: Any
    market_fill_at_open: Any
    exit_market_fill_at_open: Any
    n: int


def day(i: int) -> datetime.date:
    return datetime.date(2024, 1, 1) + datetime.timedelta(days=i)


def rows(first: int, last: int) -> tuple[datetime.date, datetime.date]:
    """data_range covering row indices [first, last] of a make_df() combo."""
    return (day(first), day(last))


def make_df(
    products: tuple[str, ...] = ("BTC",),
    resolutions: tuple[str, ...] = ("1d",),
    n: int = N,
) -> pl.DataFrame:
    """Long df sorted by date with the combos interleaved on every date.

    Per combo, bar i has close = base + i, open = close - 0.5, high/low = close ± 1.
    """
    records: list[dict[str, Any]] = []
    for i in range(n):
        for product in products:
            for resolution in resolutions:
                close = BASE_PRICE[product] + i
                records.append(
                    {
                        "date": datetime.datetime.combine(day(i), datetime.time()),
                        "product": product,
                        "resolution": resolution,
                        "open": close - 0.5,
                        "high": close + 1.0,
                        "low": close - 1.0,
                        "close": close,
                        "volume": 1000.0 + i,
                    }
                )
    return pl.DataFrame(records)


def configure(
    df: pl.DataFrame,
    product: str = "BTC",
    *,
    resolution: str = "1d",
    signal: list[int | None] | None = None,
    data_range: tuple[datetime.date, datetime.date] | None = None,
    open_kwargs: dict[str, Any] | None = None,
    close_kwargs: dict[str, Any] | None = None,
    skip_open: bool = False,
    skip_close: bool = False,
) -> None:
    """Register one segment of one combo the way a user would.

    `signal` (and any series in open_kwargs) is positional over ALL of the
    combo's rows; create_signal/open_position slice it to data_range.
    """
    sub = df.filter(
        (pl.col("product") == product) & (pl.col("resolution") == resolution)
    )
    values = [None] * len(sub) if signal is None else signal
    Mixin.create_signal(
        sub, signal=pl.Series("signal", values, dtype=pl.Int64), data_range=data_range
    )
    if not skip_open:
        Mixin.open_position(sub, **(open_kwargs or {}))
    if not skip_close:
        Mixin.close_position(sub, **(close_kwargs or {}))


def col(
    result: pl.DataFrame, name: str, product: str = "BTC", resolution: str = "1d"
) -> Any:
    return (
        result.filter(
            (pl.col("product") == product) & (pl.col("resolution") == resolution)
        )
        .get_column(name)
        .to_numpy()
    )


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
        m, "backtest_loop_kernel", side_effect=m.backtest_loop_kernel
    )
    return lambda: [KernelInputs(*call.args) for call in mock.call_args_list]


class TestErrors:
    def test_nothing_registered(self):
        with pytest.raises(ValueError, match="backtest\\(\\) cannot run"):
            Mixin.backtest(make_df())

    def test_segment_without_open_position(self):
        df = make_df()
        configure(df, signal=[1, None, None, None, None, None], skip_open=True)
        with pytest.raises(
            ValueError, match="missing open_position\\(\\)/close_position\\(\\)"
        ):
            Mixin.backtest(df)

    def test_segment_without_close_position(self):
        df = make_df()
        configure(df, signal=[1, None, None, None, None, None], skip_close=True)
        with pytest.raises(
            ValueError, match="missing open_position\\(\\)/close_position\\(\\)"
        ):
            Mixin.backtest(df)

    def test_only_the_latest_segment_missing_close_position(self):
        df = make_df()
        configure(df, data_range=rows(0, 2))
        configure(df, data_range=rows(3, 5), skip_close=True)
        with pytest.raises(
            ValueError, match="missing open_position\\(\\)/close_position\\(\\)"
        ):
            Mixin.backtest(df)

    def test_configured_combo_absent_from_the_df(self):
        configure(make_df(("BTC", "ETH")), "BTC")
        with pytest.raises(
            ValueError, match="has no rows in the dataframe passed to backtest"
        ):
            Mixin.backtest(make_df(("ETH",)))

    def test_registered_dates_missing_from_the_df(self):
        df = make_df()
        configure(df)
        shorter = df.filter(pl.col("date").dt.date() != day(3))
        with pytest.raises(
            ValueError, match="are missing from the dataframe passed to backtest"
        ):
            Mixin.backtest(shorter)

    def test_duplicate_dates_within_a_combo(self):
        df = make_df()
        configure(df)
        doubled = pl.concat([df, df]).sort("date")
        with pytest.raises(
            ValueError, match="duplicate dates in the dataframe passed to backtest"
        ):
            Mixin.backtest(doubled)

    @pytest.mark.parametrize("quantity", [0, -1, -0.5])
    def test_non_positive_scalar_order_quantity(self, quantity: float):
        df = make_df()
        configure(df, open_kwargs={"order_quantity": quantity})
        with pytest.raises(ValueError, match="'order_quantity' must be positive"):
            Mixin.backtest(df)

    def test_non_positive_order_quantity_series(self):
        df = make_df()
        configure(
            df,
            open_kwargs={"order_quantity": pl.Series([1.0, 1.0, -1.0, 1.0, 1.0, 1.0])},
        )
        with pytest.raises(
            ValueError, match="'order_quantity' values must be positive or nan"
        ):
            Mixin.backtest(df)

    def test_non_positive_order_price_series(self):
        df = make_df()
        configure(
            df,
            open_kwargs={
                "order_price": pl.Series([100.0, 0.0, 100.0, 100.0, 100.0, 100.0])
            },
        )
        with pytest.raises(ValueError, match="'order_price' must be positive or nan"):
            Mixin.backtest(df)


class TestOutputFrame:
    def test_appends_the_output_columns_and_keeps_the_rows(self):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", signal=[1, None, None, None, None, None])
        result = Mixin.backtest(df)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == [*df.columns, *OUTPUT_COLS]
        assert result.select(df.columns).equals(df)
        assert all(result.schema[c] == pl.Float64 for c in OUTPUT_COLS)

    def test_input_df_is_not_modified(self):
        df = make_df()
        before = df.clone()
        configure(df, signal=[1, None, None, None, None, None])
        Mixin.backtest(df)
        assert df.equals(before)

    def test_signal_column_echoes_the_registered_signal(self):
        df = make_df()
        configure(df, signal=[1, None, -1, None, 1, None])
        eq(col(Mixin.backtest(df), "signal"), [1, NAN, -1, NAN, 1, NAN])

    def test_unconfigured_combo_is_all_nan(self):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", signal=[1, None, None, None, None, None])
        result = Mixin.backtest(df)
        for name in OUTPUT_COLS:
            eq(col(result, name, "ETH"), [NAN] * N)
        # a configured combo reports a real (flat) position instead of nan
        eq(col(result, "position", "BTC"), [0, 1, 1, 1, 1, 1])

    def test_each_combo_gets_its_own_results_on_its_own_rows(self):
        df = make_df(("BTC", "ETH"))
        configure(
            df,
            "BTC",
            signal=[1, None, None, None, None, None],
            open_kwargs={"order_quantity": 2},
        )
        configure(
            df,
            "ETH",
            signal=[None, -1, None, None, None, None],
            open_kwargs={"order_quantity": 3},
        )
        result = Mixin.backtest(df)
        eq(col(result, "position", "BTC"), [0, 2, 2, 2, 2, 2])
        eq(col(result, "trade_price", "BTC"), [NAN, 100, NAN, NAN, NAN, NAN])
        eq(col(result, "position", "ETH"), [0, 0, -3, -3, -3, -3])
        eq(col(result, "trade_price", "ETH"), [NAN, NAN, 11, NAN, NAN, NAN])

    def test_combo_results_do_not_depend_on_the_other_combos(self):
        both = make_df(("BTC", "ETH"))
        configure(
            both,
            "BTC",
            signal=[1, None, -1, None, None, None],
            close_kwargs={"time_window": 2},
        )
        configure(both, "ETH", signal=[-1, None, None, 1, None, None])
        together = Mixin.backtest(both)

        m._clear_registry()
        alone_df = make_df(("BTC",))
        configure(
            alone_df,
            "BTC",
            signal=[1, None, -1, None, None, None],
            close_kwargs={"time_window": 2},
        )
        alone = Mixin.backtest(alone_df)
        for name in OUTPUT_COLS:
            eq(col(together, name, "BTC"), col(alone, name, "BTC"))

    def test_same_product_at_two_resolutions_is_two_combos(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC",), ("1d", "1h"))
        configure(df, resolution="1d", signal=[1, None, None, None, None, None])
        configure(df, resolution="1h", signal=[-1, None, None, None, None, None])
        result = Mixin.backtest(df)
        assert len(kernel_inputs()) == 2
        eq(col(result, "position", resolution="1d"), [0, 1, 1, 1, 1, 1])
        eq(col(result, "position", resolution="1h"), [0, -1, -1, -1, -1, -1])

    def test_rows_grouped_by_product_instead_of_interleaved(self):
        # backtest() must not assume any particular row layout across combos
        df = make_df(("BTC", "ETH")).sort("date", "product", descending=[False, True])
        configure(df, "BTC", signal=[1, None, None, None, None, None])
        configure(df, "ETH", signal=[None, None, -1, None, None, None])
        result = Mixin.backtest(df)
        eq(col(result, "position", "BTC"), [0, 1, 1, 1, 1, 1])
        eq(col(result, "position", "ETH"), [0, 0, 0, -1, -1, -1])

    def test_all_segments_empty_leaves_the_combo_unbacktested(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        configure(
            df, data_range=(datetime.date(2030, 1, 1), datetime.date(2030, 1, 31))
        )
        result = Mixin.backtest(df)
        assert kernel_inputs() == []
        for name in OUTPUT_COLS:
            eq(col(result, name), [NAN] * N)

    def test_pandas_backend_with_a_non_default_index(self):
        pl_df = make_df(("BTC", "ETH"))
        configure(pl_df, "BTC", signal=[1, None, None, -1, None, None])
        expected = Mixin.backtest(pl_df)

        m._clear_registry()
        pd_df = pl_df.to_pandas()
        pd_df.index = pd_df.index[::-1]  # labels no longer match positions
        sub = pd_df[pd_df["product"] == "BTC"]
        signal = pd.Series([1.0, NAN, NAN, -1.0, NAN, NAN], index=sub.index)
        Mixin.create_signal(sub, signal=signal)
        Mixin.open_position(sub)
        Mixin.close_position(sub)
        result = Mixin.backtest(pd_df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == expected.columns
        assert list(result.index) == list(pd_df.index)
        for name in OUTPUT_COLS:
            eq(result[name].to_numpy(), expected.get_column(name).to_numpy())


class TestKernelInputs:
    def test_one_call_per_configured_combo_with_its_own_bars(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC", "ETH"))
        configure(df, "ETH")
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        assert call.n == N
        eq(call.close, [10, 11, 12, 13, 14, 15])
        eq(call.open, [9.5, 10.5, 11.5, 12.5, 13.5, 14.5])
        eq(call.high, [11, 12, 13, 14, 15, 16])
        eq(call.low, [9, 10, 11, 12, 13, 14])
        eq(call.volume, [1000, 1001, 1002, 1003, 1004, 1005])

    def test_every_input_array_has_the_combo_length_and_dtype(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC", "ETH"))
        configure(df, "BTC", data_range=rows(1, 3))
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        bool_fields = {
            "in_universe",
            "first_only",
            "long_only",
            "short_only",
            "market_fill_at_open",
            "exit_market_fill_at_open",
        }
        for name, value in call._asdict().items():
            if name == "n":
                continue
            assert value.shape == (N,), name
            if name in bool_fields:
                assert value.dtype == np.bool_, name
            elif name == "time_window":
                assert value.dtype == np.int64, name
            else:
                assert value.dtype == np.float64, name

    def test_defaults(self, kernel_inputs: Callable[[], list[KernelInputs]]):
        df = make_df()
        configure(df, signal=[1, None, -1, None, None, None])
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        eq(call.signal, [1, NAN, -1, NAN, NAN, NAN])
        eq_bool(call.in_universe, "TTTTTT")
        # order_price=None → market orders at each bar's close
        eq(call.order_price, [100, 101, 102, 103, 104, 105])
        eq(call.order_quantity, [1] * N)
        eq_bool(call.first_only, "FFFFFF")
        eq_bool(call.long_only, "FFFFFF")
        eq_bool(call.short_only, "FFFFFF")
        eq(call.take_profit, [NAN] * N)
        eq(call.stop_loss, [NAN] * N)
        eq(call.trailing_stop, [NAN] * N)
        assert call.time_window.tolist() == [-1] * N
        # fill_price='close' → fill at the previous close, not the open
        eq_bool(call.market_fill_at_open, "FFFFFF")
        eq_bool(call.exit_market_fill_at_open, "FFFFFF")

    def test_explicit_params(self, kernel_inputs: Callable[[], list[KernelInputs]]):
        df = make_df()
        configure(
            df,
            open_kwargs={
                "order_quantity": 2.5,
                "first_only": True,
                "long_only": True,
                "fill_price": "open",
            },
            close_kwargs={
                "take_profit": 0.1,
                "stop_loss": 0.05,
                "time_window": 3,
                "fill_price": "open",
            },
        )
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        eq(call.order_quantity, [2.5] * N)
        eq_bool(call.first_only, "TTTTTT")
        eq_bool(call.long_only, "TTTTTT")
        eq_bool(call.short_only, "FFFFFF")
        eq(call.take_profit, [0.1] * N)
        eq(call.stop_loss, [0.05] * N)
        eq(call.trailing_stop, [NAN] * N)
        assert call.time_window.tolist() == [3] * N
        eq_bool(call.market_fill_at_open, "TTTTTT")
        eq_bool(call.exit_market_fill_at_open, "TTTTTT")

    def test_entry_and_exit_fill_policies_are_independent(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        configure(
            df, open_kwargs={"fill_price": "open"}, close_kwargs={"fill_price": "close"}
        )
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        eq_bool(call.market_fill_at_open, "TTTTTT")
        eq_bool(call.exit_market_fill_at_open, "FFFFFF")

    def test_short_only_and_trailing_stop(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        configure(
            df, open_kwargs={"short_only": True}, close_kwargs={"trailing_stop": 0.02}
        )
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        eq_bool(call.short_only, "TTTTTT")
        eq_bool(call.long_only, "FFFFFF")
        eq(call.trailing_stop, [0.02] * N)
        eq(call.stop_loss, [NAN] * N)

    def test_order_price_and_quantity_series_are_positional(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df(("BTC", "ETH"))
        configure(
            df,
            "ETH",
            open_kwargs={
                "order_price": pl.Series([9.0, None, 11.5, 12.0, None, 14.0]),
                "order_quantity": pl.Series([1, 2, 3, 4, 5, 6]),
            },
        )
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        eq(call.order_price, [9.0, NAN, 11.5, 12.0, NAN, 14.0])
        eq(call.order_quantity, [1, 2, 3, 4, 5, 6])


class TestSegments:
    def test_rows_before_the_first_segment_are_inert(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        configure(
            df,
            signal=[1, 1, 1, 1, 1, 1],
            data_range=rows(2, 3),
            open_kwargs={"first_only": True, "fill_price": "open"},
            close_kwargs={"stop_loss": 0.05, "time_window": 3, "fill_price": "open"},
        )
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        # instructions live only on the segment's own rows
        eq(call.signal, [NAN, NAN, 1, 1, NAN, NAN])
        eq(call.order_price, [NAN, NAN, 102, 103, NAN, NAN])
        eq(call.order_quantity, [NAN, NAN, 1, 1, NAN, NAN])
        eq_bool(call.in_universe, "FFTTFF")
        # flags/risk params start at the segment and extend forward to the end
        eq_bool(call.first_only, "FFTTTT")
        eq_bool(call.market_fill_at_open, "FFTTTT")
        eq_bool(call.exit_market_fill_at_open, "FFTTTT")
        eq(call.stop_loss, [NAN, NAN, 0.05, 0.05, 0.05, 0.05])
        assert call.time_window.tolist() == [-1, -1, 3, 3, 3, 3]

    def test_contiguous_segments_switch_params_at_the_boundary(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        signal: list[int | None] = [1, None, None, -1, None, None]
        configure(
            df,
            signal=signal,
            data_range=rows(0, 2),
            open_kwargs={"order_quantity": 1, "long_only": True, "fill_price": "open"},
            close_kwargs={"take_profit": 0.1, "time_window": 4},
        )
        configure(
            df,
            signal=signal,
            data_range=rows(3, 5),
            open_kwargs={"order_quantity": 5, "first_only": True},
            close_kwargs={"stop_loss": 0.02, "fill_price": "open"},
        )
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        eq(call.signal, [1, NAN, NAN, -1, NAN, NAN])
        eq_bool(call.in_universe, "TTTTTT")
        eq(call.order_quantity, [1, 1, 1, 5, 5, 5])
        eq(call.order_price, [100, 101, 102, 103, 104, 105])
        eq_bool(call.long_only, "TTTFFF")
        eq_bool(call.first_only, "FFFTTT")
        eq_bool(call.market_fill_at_open, "TTTFFF")
        eq_bool(call.exit_market_fill_at_open, "FFFTTT")
        # a later segment passing None genuinely disables the earlier value
        eq(call.take_profit, [0.1, 0.1, 0.1, NAN, NAN, NAN])
        eq(call.stop_loss, [NAN, NAN, NAN, 0.02, 0.02, 0.02])
        assert call.time_window.tolist() == [4, 4, 4, -1, -1, -1]

    def test_gap_rows_are_out_of_universe_and_inherit_the_previous_params(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        configure(
            df,
            signal=[1, 1, 1, 1, 1, 1],
            data_range=rows(0, 1),
            open_kwargs={"long_only": True},
            close_kwargs={"take_profit": 0.1},
        )
        configure(
            df,
            signal=[1, 1, 1, 1, 1, 1],
            data_range=rows(4, 5),
            open_kwargs={"short_only": True},
        )
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        eq_bool(call.in_universe, "TTFFTT")
        eq(call.signal, [1, 1, NAN, NAN, 1, 1])
        eq(call.order_price, [100, 101, NAN, NAN, 104, 105])
        eq(call.order_quantity, [1, 1, NAN, NAN, 1, 1])
        eq_bool(call.long_only, "TTTTFF")
        eq_bool(call.short_only, "FFFFTT")
        eq(call.take_profit, [0.1, 0.1, 0.1, 0.1, NAN, NAN])

    def test_registration_order_does_not_matter(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        signal: list[int | None] = [1, None, None, -1, None, 1]
        early: dict[str, Any] = {
            "data_range": rows(0, 2),
            "open_kwargs": {"order_quantity": 2, "fill_price": "open"},
            "close_kwargs": {"stop_loss": 0.03},
        }
        late: dict[str, Any] = {
            "data_range": rows(3, 5),
            "open_kwargs": {"order_quantity": 7, "first_only": True},
            "close_kwargs": {"time_window": 2},
        }
        df = make_df()
        configure(df, signal=signal, **early)
        configure(df, signal=signal, **late)
        forward = Mixin.backtest(df)

        m._clear_registry()
        configure(df, signal=signal, **late)
        configure(df, signal=signal, **early)
        backward = Mixin.backtest(df)

        first, second = kernel_inputs()
        for name, value in first._asdict().items():
            np.testing.assert_array_equal(value, getattr(second, name), err_msg=name)
        assert forward.equals(backward, null_equal=True)

    def test_empty_segment_is_skipped(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        df = make_df()
        configure(df, signal=[1, None, None, None, None, None], data_range=rows(0, 5))
        configure(
            df,
            data_range=(datetime.date(2030, 1, 1), datetime.date(2030, 1, 31)),
            close_kwargs={"stop_loss": 0.5},
        )
        result = Mixin.backtest(df)
        (call,) = kernel_inputs()
        eq(call.stop_loss, [NAN] * N)
        eq_bool(call.in_universe, "TTTTTT")
        eq(col(result, "position"), [0, 1, 1, 1, 1, 1])

    def test_segment_registered_on_an_expanding_window(
        self, kernel_inputs: Callable[[], list[KernelInputs]]
    ):
        # the dynamic-period pattern: each period is configured on the history
        # known so far, then ONE backtest() runs over the full df
        df = make_df(("BTC", "ETH"))
        for first, last, qty in ((0, 1, 1), (2, 3, 2), (4, 5, 3)):
            window = df.filter(pl.col("date").dt.date() <= day(last))
            n_rows = len(window.filter(pl.col("product") == "BTC"))
            configure(
                window,
                "BTC",
                signal=[1] * n_rows,
                data_range=rows(first, last),
                open_kwargs={"order_quantity": qty},
            )
        Mixin.backtest(df)
        (call,) = kernel_inputs()
        eq(call.signal, [1] * N)
        eq_bool(call.in_universe, "TTTTTT")
        eq(call.order_quantity, [1, 1, 2, 2, 3, 3])
        eq(call.order_price, [100, 101, 102, 103, 104, 105])


class TestEndToEnd:
    """backtest() + the real kernel, on the behaviours that depend on both."""

    def test_position_carries_across_contiguous_segments(self):
        df = make_df()
        signal: list[int | None] = [1, None, None, None, None, None]
        configure(df, signal=signal, data_range=rows(0, 2))
        configure(df, signal=signal, data_range=rows(3, 5))
        result = Mixin.backtest(df)
        eq(col(result, "position"), [0, 1, 1, 1, 1, 1])
        eq(col(result, "trade_size"), [NAN, 1, NAN, NAN, NAN, NAN])

    def test_position_is_closed_at_the_close_of_the_first_gap_row(self):
        df = make_df()
        signal: list[int | None] = [1, None, None, None, None, None]
        configure(df, signal=signal, data_range=rows(0, 1))
        configure(df, signal=signal, data_range=rows(4, 5))
        result = Mixin.backtest(df)
        eq(col(result, "position"), [0, 1, 0, 0, 0, 0])
        eq(col(result, "trade_price"), [NAN, 100, 102, NAN, NAN, NAN])
        eq(col(result, "trade_size"), [NAN, 1, -1, NAN, NAN, NAN])

    def test_position_is_closed_when_the_last_segment_ends_early(self):
        df = make_df()
        configure(df, signal=[1, None, None, None, None, None], data_range=rows(0, 2))
        result = Mixin.backtest(df)
        eq(col(result, "position"), [0, 1, 1, 0, 0, 0])
        eq(col(result, "trade_price"), [NAN, 100, NAN, 103, NAN, NAN])

    @pytest.mark.parametrize(
        ("placing_fill", "next_fill", "expected"),
        [("open", "close", 102.5), ("close", "open", 102.0)],
    )
    def test_order_placed_on_a_segments_last_bar_uses_that_segments_fill_policy(
        self, placing_fill: str, next_fill: str, expected: float
    ):
        df = make_df()
        signal: list[int | None] = [None, None, 1, None, None, None]
        configure(
            df,
            signal=signal,
            data_range=rows(0, 2),
            open_kwargs={"fill_price": placing_fill},
        )
        configure(
            df,
            signal=signal,
            data_range=rows(3, 5),
            open_kwargs={"fill_price": next_fill},
        )
        result = Mixin.backtest(df)
        eq(col(result, "trade_price"), [NAN, NAN, NAN, expected, NAN, NAN])

    def test_time_window_applies_only_while_its_segment_governs(self):
        # time_window=4 would fire at bar 4, but the second segment (from row 3)
        # passes None and disables it first, so the position survives
        df = make_df()
        signal: list[int | None] = [1, None, None, None, None, None]
        configure(
            df, signal=signal, data_range=rows(0, 2), close_kwargs={"time_window": 4}
        )
        configure(df, signal=signal, data_range=rows(3, 5))
        eq(col(Mixin.backtest(df), "position"), [0, 1, 1, 1, 1, 1])

        m._clear_registry()
        configure(
            df, signal=signal, data_range=rows(0, 2), close_kwargs={"time_window": 2}
        )
        configure(df, signal=signal, data_range=rows(3, 5))
        # held bars 1-2 → close order placed at bar 2 (still segment 1) → fills at bar 3
        eq(col(Mixin.backtest(df), "position"), [0, 1, 1, 0, 0, 0])
