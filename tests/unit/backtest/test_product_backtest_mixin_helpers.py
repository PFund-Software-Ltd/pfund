"""Tests the module-level helper functions of product_backtest_mixin directly
(the functions outside ProductBacktestMixin)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

import datetime
import importlib
import importlib.util
import sys
from pathlib import Path

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest


def _load_mixin_module() -> ModuleType:
    name = "pfund.backtest._mixins.product_backtest_mixin"
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

NAN = np.nan
D = datetime.date
DT = datetime.datetime


def day(i: int) -> datetime.date:
    return D(2024, 1, 1) + datetime.timedelta(days=i)


def midnight(i: int) -> datetime.datetime:
    return DT.combine(day(i), datetime.time())


def make_df(
    n: int = 5,
    products: tuple[str, ...] = ("BTC",),
    resolutions: tuple[str, ...] = ("1d",),
) -> Any:
    """narwhals df, one row per (date, product, resolution), sorted by date."""
    records = [
        {"date": midnight(i), "product": product, "resolution": resolution}
        for i in range(n)
        for product in products
        for resolution in resolutions
    ]
    return nw.from_native(pl.DataFrame(records), eager_only=True)


def dates_df(dates: list[datetime.datetime]) -> Any:
    return nw.from_native(pl.DataFrame({"date": dates}), eager_only=True)


def intraday(*stamps: str) -> np.ndarray:
    return np.array(stamps, dtype="datetime64[us]")


def eq(actual: Any, expected: Any) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        equal_nan=True,
    )


@pytest.fixture(autouse=True)
def fresh_registry() -> Iterator[None]:
    m._clear_registry()
    yield
    m._clear_registry()


class TestClearRegistry:
    def test_empties_the_registry_and_the_latest_key_map(self):
        key = ((day(0), day(1)), "BTC", "1d")
        m._registry[key] = {"signal": np.array([1.0])}
        m._latest_key_by_combo[("BTC", "1d")] = key
        m._clear_registry()
        assert m._registry == {}
        assert m._latest_key_by_combo == {}

    def test_clears_in_place_so_existing_references_see_it(self):
        registry, latest = m._registry, m._latest_key_by_combo
        registry["k"] = {}
        latest["c"] = "k"
        m._clear_registry()
        assert registry == {} and latest == {}
        assert m._registry is registry
        assert m._latest_key_by_combo is latest


class TestGetCurrentCombo:
    def test_returns_product_then_resolution(self):
        assert m._get_current_combo(make_df()) == ("BTC", "1d")

    def test_does_not_depend_on_the_column_order_of_the_df(self):
        df = make_df().select("resolution", "date", "product")
        assert m._get_current_combo(df) == ("BTC", "1d")

    def test_more_than_one_product_raises(self):
        with pytest.raises(ValueError, match="Multiple \\(product, resolution\\)s"):
            m._get_current_combo(make_df(products=("BTC", "ETH")))

    def test_more_than_one_resolution_raises(self):
        with pytest.raises(ValueError, match="Multiple \\(product, resolution\\)s"):
            m._get_current_combo(make_df(resolutions=("1d", "1h")))

    def test_error_shows_the_usage_pattern(self):
        with pytest.raises(ValueError) as exc_info:
            m._get_current_combo(make_df(products=("BTC", "ETH")))
        assert m._PATTERN in str(exc_info.value)


class TestGetRegistryKey:
    def test_explicit_data_range_is_kept_as_passed(self):
        data_range = (day(1), day(2))
        key = m._get_registry_key(make_df(), data_range)
        assert key == (data_range, "BTC", "1d")
        assert key[0] is data_range

    def test_none_becomes_the_first_and_last_dates_of_the_df(self):
        key = m._get_registry_key(make_df(n=5), None)
        assert key == ((midnight(0), midnight(4)), "BTC", "1d")

    def test_single_row_df_has_equal_bounds(self):
        key = m._get_registry_key(make_df(n=1), None)
        assert key == ((midnight(0), midnight(0)), "BTC", "1d")

    def test_same_combo_and_range_give_the_same_key(self):
        data_range = (day(0), day(1))
        assert m._get_registry_key(make_df(), data_range) == m._get_registry_key(
            make_df(n=3), data_range
        )

    def test_multi_combo_df_raises(self):
        with pytest.raises(ValueError, match="Multiple \\(product, resolution\\)s"):
            m._get_registry_key(make_df(products=("BTC", "ETH")), None)


class TestValidateDataRange:
    @pytest.mark.parametrize(
        "data_range",
        [
            (D(2024, 1, 1), D(2024, 1, 31)),
            (DT(2024, 1, 1, 9), DT(2024, 1, 1, 17)),
            (D(2024, 1, 1), DT(2024, 1, 1, 17)),
            (DT(2024, 1, 1, 9), D(2024, 1, 2)),
        ],
    )
    def test_valid_range_is_returned_unchanged(self, data_range: tuple[Any, Any]):
        assert m._validate_data_range(data_range) == data_range

    def test_start_equal_to_end_is_valid(self):
        assert m._validate_data_range((day(0), day(0))) == (day(0), day(0))

    @pytest.mark.parametrize(
        "data_range",
        [
            [D(2024, 1, 1), D(2024, 1, 2)],
            (D(2024, 1, 1),),
            (D(2024, 1, 1), D(2024, 1, 2), D(2024, 1, 3)),
            D(2024, 1, 1),
            "2024-01-01",
        ],
    )
    def test_not_a_two_tuple_raises(self, data_range: Any):
        with pytest.raises(ValueError, match="must be a \\(start_date, end_date\\)"):
            m._validate_data_range(data_range)

    @pytest.mark.parametrize(
        ("data_range", "type_name"),
        [
            (("2024-01-01", D(2024, 1, 2)), "str"),
            ((D(2024, 1, 1), "2024-01-02"), "str"),
            ((D(2024, 1, 1), None), "NoneType"),
            ((np.datetime64("2024-01-01"), D(2024, 1, 2)), "datetime64"),
            ((20240101, D(2024, 1, 2)), "int"),
        ],
    )
    def test_non_date_bound_raises(self, data_range: Any, type_name: str):
        with pytest.raises(
            ValueError, match=f"bounds must be date/datetime, got {type_name}"
        ):
            m._validate_data_range(data_range)

    @pytest.mark.parametrize(
        "data_range",
        [
            (D(2024, 1, 2), D(2024, 1, 1)),
            (DT(2024, 1, 1, 17), DT(2024, 1, 1, 9)),
            (D(2024, 1, 2), DT(2024, 1, 1, 23)),
            (DT(2024, 1, 1, 0, 0, 1), D(2024, 1, 1)),
        ],
    )
    def test_start_after_end_raises(self, data_range: tuple[Any, Any]):
        with pytest.raises(ValueError, match="is after its end"):
            m._validate_data_range(data_range)


class TestDataRangeMask:
    BARS = intraday(
        "2024-01-01T00:00",
        "2024-01-01T12:00",
        "2024-01-02T00:00",
        "2024-01-02T23:59:59.999999",
        "2024-01-03T00:00",
    )

    def test_both_bounds_are_inclusive(self):
        mask = m._data_range_mask(self.BARS, DT(2024, 1, 1, 12), DT(2024, 1, 2))
        assert mask.tolist() == [False, True, True, False, False]

    def test_plain_date_bounds_cover_whole_days(self):
        mask = m._data_range_mask(self.BARS, D(2024, 1, 2), D(2024, 1, 2))
        assert mask.tolist() == [False, False, True, True, False]

    def test_plain_date_start_includes_the_days_first_bar(self):
        mask = m._data_range_mask(self.BARS, D(2024, 1, 1), DT(2024, 1, 1, 12))
        assert mask.tolist() == [True, True, False, False, False]

    def test_plain_date_end_includes_the_days_last_bar(self):
        mask = m._data_range_mask(self.BARS, DT(2024, 1, 1, 12), D(2024, 1, 2))
        assert mask.tolist() == [False, True, True, True, False]

    def test_datetime_bounds_compare_exactly(self):
        mask = m._data_range_mask(
            self.BARS, DT(2024, 1, 1, 0, 0, 1), DT(2024, 1, 2, 23, 59, 59)
        )
        assert mask.tolist() == [False, True, True, False, False]

    def test_midnight_datetime_end_excludes_the_rest_of_that_day(self):
        mask = m._data_range_mask(self.BARS, D(2024, 1, 1), DT(2024, 1, 2))
        assert mask.tolist() == [True, True, True, False, False]

    def test_range_outside_the_data_selects_nothing(self):
        mask = m._data_range_mask(self.BARS, D(2025, 1, 1), D(2025, 12, 31))
        assert not mask.any()

    def test_range_wider_than_the_data_selects_everything(self):
        mask = m._data_range_mask(self.BARS, D(2000, 1, 1), D(2100, 1, 1))
        assert mask.all()

    def test_empty_date_array(self):
        mask = m._data_range_mask(intraday(), D(2024, 1, 1), D(2024, 1, 2))
        assert mask.dtype == np.bool_
        assert len(mask) == 0

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "D"])
    def test_any_datetime64_unit(self, unit: str):
        bars = np.array(["2024-01-01", "2024-01-02", "2024-01-03"]).astype(
            f"datetime64[{unit}]"
        )
        assert m._data_range_mask(bars, D(2024, 1, 2), D(2024, 1, 3)).tolist() == [
            False,
            True,
            True,
        ]
        assert m._data_range_mask(bars, DT(2024, 1, 2), DT(2024, 1, 3)).tolist() == [
            False,
            True,
            True,
        ]

    def test_does_not_modify_the_date_array(self):
        bars = self.BARS.copy()
        m._data_range_mask(bars, D(2024, 1, 2), D(2024, 1, 2))
        assert (bars == self.BARS).all()
        assert bars.dtype == self.BARS.dtype


class TestToFloat64:
    @staticmethod
    def _convert(native: Any) -> np.ndarray:
        return m._to_float64(nw.from_native(native, series_only=True))

    @pytest.mark.parametrize(
        "native",
        [
            pl.Series("x", [1, 2, 3], dtype=pl.Int64),
            pl.Series("x", [1.0, 2.0, 3.0], dtype=pl.Float32),
            pl.Series("x", [1.0, 2.0, 3.0], dtype=pl.Float64),
            pd.Series([1, 2, 3], dtype="int64"),
            pd.Series([1.0, 2.0, 3.0], dtype="float64"),
        ],
    )
    def test_result_is_a_float64_numpy_array(self, native: Any):
        arr = self._convert(native)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        eq(arr, [1, 2, 3])

    @pytest.mark.parametrize(
        "native",
        [
            pl.Series("x", [1, None, 3], dtype=pl.Int64),
            pl.Series("x", [1.0, None, 3.0], dtype=pl.Float64),
            pl.Series("x", [1.0, NAN, 3.0], dtype=pl.Float64),
            pd.Series([1.0, None, 3.0], dtype="float64"),
            pd.Series([1, None, 3], dtype="Int64"),
        ],
    )
    def test_missing_values_become_nan(self, native: Any):
        eq(self._convert(native), [1, NAN, 3])

    def test_bool_becomes_one_and_zero(self):
        eq(self._convert(pl.Series("x", [True, False])), [1, 0])

    def test_empty_series(self):
        arr = self._convert(pl.Series("x", [], dtype=pl.Float64))
        assert arr.dtype == np.float64
        assert len(arr) == 0

    def test_pandas_values_are_positional_not_index_ordered(self):
        eq(self._convert(pd.Series([1.0, 2.0, 3.0], index=[2, 0, 1])), [1, 2, 3])


class TestSliceSeriesToDataRange:
    @pytest.mark.parametrize("value", [None, 2, 2.5, np.float64(2.5), np.int64(2)])
    def test_none_and_scalars_pass_through(self, value: Any):
        result = m._slice_series_to_data_range(
            make_df(), (day(1), day(2)), value, "order_quantity"
        )
        assert result is value

    def test_scalar_passes_through_without_validating_the_range(self):
        assert m._slice_series_to_data_range(make_df(), "not a range", 2, "x") == 2

    def test_polars_series_is_sliced_to_the_range_rows(self):
        series = pl.Series("order_price", [10.0, 11.0, 12.0, 13.0, 14.0])
        result = m._slice_series_to_data_range(
            make_df(n=5), (day(1), day(3)), series, "order_price"
        )
        assert isinstance(result, pl.Series)
        assert result.to_list() == [11.0, 12.0, 13.0]
        assert result.name == "order_price"

    def test_input_series_is_not_modified(self):
        series = pl.Series("order_price", [10.0, 11.0, 12.0, 13.0, 14.0])
        m._slice_series_to_data_range(
            make_df(n=5), (day(1), day(3)), series, "order_price"
        )
        assert series.to_list() == [10.0, 11.0, 12.0, 13.0, 14.0]

    def test_pandas_series_is_sliced_by_position_not_by_index(self):
        series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=[4, 3, 2, 1, 0])
        result = m._slice_series_to_data_range(
            make_df(n=5), (day(1), day(3)), series, "order_price"
        )
        assert isinstance(result, pd.Series)
        assert result.tolist() == [11.0, 12.0, 13.0]

    def test_range_covering_everything_keeps_every_row(self):
        series = pl.Series("x", [10.0, 11.0, 12.0])
        result = m._slice_series_to_data_range(
            make_df(n=3), (day(0), day(2)), series, "x"
        )
        assert result.to_list() == [10.0, 11.0, 12.0]

    def test_range_covering_nothing_gives_an_empty_series(self):
        series = pl.Series("x", [10.0, 11.0, 12.0])
        result = m._slice_series_to_data_range(
            make_df(n=3), (day(10), day(20)), series, "x"
        )
        assert len(result) == 0
        assert result.dtype == pl.Float64

    def test_nulls_inside_the_range_are_kept(self):
        series = pl.Series("x", [10.0, None, 12.0])
        result = m._slice_series_to_data_range(
            make_df(n=3), (day(1), day(2)), series, "x"
        )
        assert result.to_list() == [None, 12.0]

    @pytest.mark.parametrize("length", [0, 4, 6])
    def test_series_not_matching_the_df_length_raises(self, length: int):
        series = pl.Series("order_price", [1.0] * length)
        with pytest.raises(
            ValueError,
            match=f"'order_price' length \\({length}\\) must match the dataframe "
            "length \\(5\\)",
        ):
            m._slice_series_to_data_range(
                make_df(n=5), (day(1), day(3)), series, "order_price"
            )

    def test_invalid_data_range_raises_for_a_series(self):
        series = pl.Series("x", [1.0] * 5)
        with pytest.raises(ValueError, match="is after its end"):
            m._slice_series_to_data_range(make_df(n=5), (day(3), day(1)), series, "x")


class TestGroupPositions:
    def test_empty_input_gives_no_groups(self):
        empty = np.array([], dtype=object)
        assert m._group_positions([empty, empty], 0) == {}

    def test_interleaved_rows_are_grouped_per_combo(self):
        product = np.array(["BTC", "ETH", "BTC", "ETH", "BTC", "ETH"])
        resolution = np.array(["1d"] * 6)
        groups = m._group_positions([product, resolution], 6)
        assert set(groups) == {("BTC", "1d"), ("ETH", "1d")}
        assert groups[("BTC", "1d")].tolist() == [0, 2, 4]
        assert groups[("ETH", "1d")].tolist() == [1, 3, 5]

    def test_blocked_rows_are_grouped_per_combo(self):
        product = np.array(["ETH", "ETH", "BTC", "BTC", "BTC"])
        resolution = np.array(["1d"] * 5)
        groups = m._group_positions([product, resolution], 5)
        assert groups[("ETH", "1d")].tolist() == [0, 1]
        assert groups[("BTC", "1d")].tolist() == [2, 3, 4]

    def test_same_product_at_two_resolutions_is_two_groups(self):
        product = np.array(["BTC", "BTC", "BTC", "BTC"])
        resolution = np.array(["1d", "1h", "1d", "1h"])
        groups = m._group_positions([product, resolution], 4)
        assert groups[("BTC", "1d")].tolist() == [0, 2]
        assert groups[("BTC", "1h")].tolist() == [1, 3]

    def test_same_resolution_for_two_products_is_two_groups(self):
        product = np.array(["BTC", "ETH", "ETH", "BTC"])
        resolution = np.array(["1h", "1h", "1h", "1h"])
        groups = m._group_positions([product, resolution], 4)
        assert groups[("BTC", "1h")].tolist() == [0, 3]
        assert groups[("ETH", "1h")].tolist() == [1, 2]

    def test_positions_are_ascending_and_cover_every_row_once(self):
        rng = np.random.default_rng(0)
        n = 500
        product = rng.choice(np.array(["BTC", "ETH", "SOL"]), n)
        resolution = rng.choice(np.array(["1d", "1h"]), n)
        groups = m._group_positions([product, resolution], n)
        for (p, r), pos in groups.items():
            assert (np.diff(pos) > 0).all()
            assert (product[pos] == p).all()
            assert (resolution[pos] == r).all()
            assert len(pos) == ((product == p) & (resolution == r)).sum()
        assert sorted(np.concatenate(list(groups.values())).tolist()) == list(range(n))

    def test_single_combo_gets_every_row(self):
        groups = m._group_positions([np.array(["BTC"] * 3), np.array(["1d"] * 3)], 3)
        assert list(groups) == [("BTC", "1d")]
        assert groups[("BTC", "1d")].tolist() == [0, 1, 2]

    def test_single_pivot_column_gives_one_element_keys(self):
        groups = m._group_positions([np.array(["ETH", "BTC", "ETH"])], 3)
        assert groups[("BTC",)].tolist() == [1]
        assert groups[("ETH",)].tolist() == [0, 2]

    def test_groups_are_ordered_by_the_first_pivot_column_first(self):
        product = np.array(["ETH", "BTC", "ETH", "BTC"])
        resolution = np.array(["1d", "1h", "1h", "1d"])
        groups = m._group_positions([product, resolution], 4)
        assert list(groups) == [
            ("BTC", "1d"),
            ("BTC", "1h"),
            ("ETH", "1d"),
            ("ETH", "1h"),
        ]

    def test_keys_are_python_values_for_numpy_string_arrays(self):
        groups = m._group_positions([np.array(["BTC"]), np.array(["1d"])], 1)
        (key,) = groups
        assert [type(v) for v in key] == [str, str]

    def test_keys_are_python_values_for_numeric_arrays(self):
        groups = m._group_positions([np.array([7, 7, 3], dtype=np.int64)], 3)
        assert all(type(k[0]) is int for k in groups)
        assert groups[(3,)].tolist() == [2]
        assert groups[(7,)].tolist() == [0, 1]

    @pytest.mark.parametrize("backend", ["polars", "pandas"])
    def test_keys_match_the_combo_read_from_the_dataframe(self, backend: str):
        pl_df = pl.DataFrame(
            {"product": ["BTC", "ETH", "BTC"], "resolution": ["1d", "1d", "1d"]}
        )
        df = nw.from_native(
            pl_df if backend == "polars" else pl_df.to_pandas(), eager_only=True
        )
        groups = m._group_positions(
            [df.get_column(c).to_numpy() for c in m.PIVOT_COLS], len(df)
        )
        assert set(groups) == set(df.select(*m.PIVOT_COLS).unique().rows())
        assert groups[("BTC", "1d")].tolist() == [0, 2]

    def test_does_not_modify_the_input_arrays(self):
        product = np.array(["ETH", "BTC", "ETH"])
        resolution = np.array(["1d", "1d", "1d"])
        m._group_positions([product, resolution], 3)
        assert product.tolist() == ["ETH", "BTC", "ETH"]
        assert resolution.tolist() == ["1d", "1d", "1d"]


class TestSeriesToPositionalFloat64:
    def test_polars_series(self):
        arr = m._series_to_positional_float64(pl.Series("x", [1, 2, 3]), 3, "x")
        assert arr.dtype == np.float64
        eq(arr, [1, 2, 3])

    def test_pandas_series(self):
        arr = m._series_to_positional_float64(pd.Series([1, 2, 3]), 3, "x")
        assert arr.dtype == np.float64
        eq(arr, [1, 2, 3])

    def test_pandas_index_is_ignored(self):
        series = pd.Series([1.0, 2.0, 3.0], index=["c", "a", "b"])
        eq(m._series_to_positional_float64(series, 3, "x"), [1, 2, 3])

    def test_nulls_become_nan(self):
        series = pl.Series("x", [1.0, None, 3.0])
        eq(m._series_to_positional_float64(series, 3, "x"), [1, NAN, 3])

    def test_empty_series_with_zero_expected_length(self):
        arr = m._series_to_positional_float64(
            pl.Series("x", [], dtype=pl.Float64), 0, "x"
        )
        assert len(arr) == 0

    @pytest.mark.parametrize(("length", "n"), [(2, 3), (4, 3), (0, 3), (3, 0)])
    def test_length_mismatch_raises(self, length: int, n: int):
        with pytest.raises(
            ValueError,
            match=f"'order_price' length \\({length}\\) does not match dataframe "
            f"length \\({n}\\)",
        ):
            m._series_to_positional_float64(
                pl.Series("x", [1.0] * length), n, "order_price"
            )

    def test_not_a_series_raises(self):
        with pytest.raises(TypeError):
            m._series_to_positional_float64([1.0, 2.0, 3.0], 3, "x")


class TestScatterOrderInputs:
    CLOSE = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

    def _scatter(
        self, seg_rows: list[int], order_price: Any = None, order_quantity: Any = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        price = np.full(len(self.CLOSE), NAN)
        quantity = np.full(len(self.CLOSE), NAN)
        m._scatter_order_inputs(
            {"order_price": order_price, "order_quantity": order_quantity},
            np.array(seg_rows, dtype=np.int64),
            self.CLOSE,
            price,
            quantity,
        )
        return price, quantity

    def test_no_order_price_uses_the_close_of_the_segment_rows(self):
        price, _ = self._scatter([1, 2, 3])
        eq(price, [NAN, 101, 102, 103, NAN])

    def test_order_price_series_is_positional_over_the_segment_rows(self):
        price, _ = self._scatter([1, 2, 3], pl.Series("p", [9.0, 8.0, 7.0]))
        eq(price, [NAN, 9, 8, 7, NAN])

    def test_nan_and_null_order_prices_stay_nan(self):
        price, _ = self._scatter([0, 1, 2], pl.Series("p", [9.0, None, NAN]))
        eq(price, [9, NAN, NAN, NAN, NAN])

    def test_pandas_order_price_series_ignores_its_index(self):
        series = pd.Series([9.0, 8.0, 7.0], index=[30, 10, 20])
        price, _ = self._scatter([1, 2, 3], series)
        eq(price, [NAN, 9, 8, 7, NAN])

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_order_price_raises(self, bad: float):
        with pytest.raises(ValueError, match="'order_price' must be positive or nan"):
            self._scatter([1, 2, 3], pl.Series("p", [9.0, bad, 7.0]))

    def test_order_price_series_of_the_wrong_length_raises(self):
        with pytest.raises(
            ValueError, match="'order_price' length \\(2\\) does not match"
        ):
            self._scatter([1, 2, 3], pl.Series("p", [9.0, 8.0]))

    @pytest.mark.parametrize("quantity", [2, 2.5, np.float64(2.5), np.int64(2)])
    def test_scalar_order_quantity_is_broadcast(self, quantity: Any):
        _, result = self._scatter([1, 2, 3], order_quantity=quantity)
        eq(result, [NAN, quantity, quantity, quantity, NAN])
        assert result.dtype == np.float64

    @pytest.mark.parametrize("quantity", [0, -1, 0.0, -0.5, NAN])
    def test_non_positive_scalar_order_quantity_raises(self, quantity: float):
        with pytest.raises(ValueError, match="'order_quantity' must be positive"):
            self._scatter([1, 2, 3], order_quantity=quantity)

    def test_order_quantity_series_is_positional_over_the_segment_rows(self):
        _, quantity = self._scatter(
            [1, 2, 3], order_quantity=pl.Series("q", [1.0, 2.0, 3.0])
        )
        eq(quantity, [NAN, 1, 2, 3, NAN])

    def test_nan_and_null_order_quantities_stay_nan(self):
        _, quantity = self._scatter(
            [0, 1, 2], order_quantity=pl.Series("q", [1.0, None, NAN])
        )
        eq(quantity, [1, NAN, NAN, NAN, NAN])

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_order_quantity_in_a_series_raises(self, bad: float):
        with pytest.raises(
            ValueError, match="'order_quantity' values must be positive or nan"
        ):
            self._scatter([1, 2, 3], order_quantity=pl.Series("q", [1.0, bad, 3.0]))

    def test_order_quantity_series_of_the_wrong_length_raises(self):
        with pytest.raises(
            ValueError, match="'order_quantity' length \\(4\\) does not match"
        ):
            self._scatter([1, 2, 3], order_quantity=pl.Series("q", [1.0] * 4))

    def test_non_contiguous_segment_rows(self):
        price, quantity = self._scatter([0, 2, 4], pl.Series("p", [9.0, 8.0, 7.0]), 3)
        eq(price, [9, NAN, 8, NAN, 7])
        eq(quantity, [3, NAN, 3, NAN, 3])

    def test_rows_outside_the_segment_are_left_as_they_were(self):
        price = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        quantity = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        m._scatter_order_inputs(
            {"order_price": None, "order_quantity": 2},
            np.array([1, 2], dtype=np.int64),
            self.CLOSE,
            price,
            quantity,
        )
        eq(price, [1, 101, 102, 4, 5])
        eq(quantity, [6, 2, 2, 9, 10])

    def test_empty_segment_writes_nothing(self):
        price, quantity = self._scatter([])
        assert np.isnan(price).all()
        assert np.isnan(quantity).all()

    def test_close_array_is_not_modified(self):
        close = self.CLOSE.copy()
        self._scatter([1, 2, 3], pl.Series("p", [9.0, 8.0, 7.0]))
        eq(self.CLOSE, close)


class TestResolveTargets:
    COMBO = ("BTC", "1d")

    @staticmethod
    def _key_identity(key: tuple[Any, ...]) -> tuple[Any, ...]:
        return key[1:]

    def _resolve(
        self,
        df: Any,
        data_range: Any,
        registry: dict[tuple[Any, ...], dict[str, Any]] | None = None,
        identity: Any = COMBO,
    ) -> tuple[np.ndarray, np.ndarray]:
        return m._resolve_targets(
            {} if registry is None else registry,
            identity,
            self._key_identity,
            df,
            data_range,
        )

    def _registered(
        self, df: Any, data_range: Any, identity: tuple[str, str] = COMBO
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        """A registry holding one entry for `identity` over `data_range`."""
        _, dates = self._resolve(df, data_range, identity=identity)
        return {(data_range, *identity): {"dates": dates}}

    def test_returns_the_positions_and_dates_inside_the_range(self):
        df = make_df(n=5)
        pos, dates = self._resolve(df, (day(1), day(3)))
        assert pos.tolist() == [1, 2, 3]
        assert (dates == df.get_column("date").to_numpy()[1:4]).all()

    def test_range_bounds_are_inclusive(self):
        pos, _ = self._resolve(make_df(n=5), (day(0), day(4)))
        assert pos.tolist() == [0, 1, 2, 3, 4]

    def test_single_day_range(self):
        pos, _ = self._resolve(make_df(n=5), (day(2), day(2)))
        assert pos.tolist() == [2]

    def test_range_outside_the_df_gives_empty_targets(self):
        pos, dates = self._resolve(make_df(n=5), (day(10), day(20)))
        assert len(pos) == 0
        assert len(dates) == 0

    def test_range_partly_outside_the_df_is_clipped_to_its_rows(self):
        pos, _ = self._resolve(make_df(n=5), (day(3), day(20)))
        assert pos.tolist() == [3, 4]

    def test_does_not_write_to_the_registry(self):
        registry: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._resolve(make_df(n=5), (day(1), day(3)), registry)
        assert registry == {}

    def test_missing_date_column_raises(self):
        df = make_df(n=5).drop("date")
        with pytest.raises(ValueError, match="requires a 'date' column"):
            self._resolve(df, (day(1), day(3)))

    def test_unsorted_dates_raise(self):
        df = dates_df([midnight(0), midnight(2), midnight(1)])
        with pytest.raises(ValueError, match="must be sorted by 'date'"):
            self._resolve(df, (day(0), day(2)))

    def test_descending_dates_raise(self):
        df = dates_df([midnight(2), midnight(1), midnight(0)])
        with pytest.raises(ValueError, match="must be sorted by 'date'"):
            self._resolve(df, (day(0), day(2)))

    def test_invalid_data_range_raises(self):
        with pytest.raises(ValueError, match="is after its end"):
            self._resolve(make_df(n=5), (day(3), day(1)))

    def test_duplicate_dates_inside_the_range_raise(self):
        df = dates_df([midnight(0), midnight(1), midnight(1), midnight(2)])
        with pytest.raises(
            ValueError, match="duplicate dates in the registered rows"
        ) as exc_info:
            self._resolve(df, (day(0), day(2)))
        assert str(exc_info.value).startswith("('BTC', '1d'): ")

    def test_duplicate_dates_outside_the_range_are_allowed(self):
        df = dates_df([midnight(0), midnight(0), midnight(1), midnight(2)])
        pos, _ = self._resolve(df, (day(1), day(2)))
        assert pos.tolist() == [2, 3]

    def test_overlap_with_a_registered_range_of_the_same_identity_raises(self):
        df = make_df(n=5)
        registry = self._registered(df, (day(0), day(2)))
        with pytest.raises(ValueError, match="a row can only be registered once"):
            self._resolve(df, (day(2), day(4)), registry)

    def test_overlap_error_names_both_ranges(self):
        df = make_df(n=5)
        registry = self._registered(df, (day(0), day(2)))
        with pytest.raises(ValueError) as exc_info:
            self._resolve(df, (day(2), day(4)), registry)
        message = str(exc_info.value)
        assert message.startswith("('BTC', '1d'): ")
        assert str((day(2), day(4))) in message
        assert str((day(0), day(2))) in message

    def test_identical_range_registered_twice_raises(self):
        df = make_df(n=5)
        registry = self._registered(df, (day(0), day(2)))
        with pytest.raises(ValueError, match="a row can only be registered once"):
            self._resolve(df, (day(0), day(2)), registry)

    def test_range_nested_inside_a_registered_one_raises(self):
        df = make_df(n=5)
        registry = self._registered(df, (day(0), day(4)))
        with pytest.raises(ValueError, match="a row can only be registered once"):
            self._resolve(df, (day(2), day(2)), registry)

    def test_adjacent_range_of_the_same_identity_is_allowed(self):
        df = make_df(n=5)
        registry = self._registered(df, (day(0), day(2)))
        pos, _ = self._resolve(df, (day(3), day(4)), registry)
        assert pos.tolist() == [3, 4]

    def test_ranges_overlapping_only_where_the_df_has_no_rows_are_allowed(self):
        # the overlap check is on registered ROWS, not on the calendar ranges
        df = dates_df([midnight(0), midnight(1), midnight(5), midnight(6)])
        registry = self._registered(df, (day(0), day(3)))
        pos, _ = self._resolve(df, (day(2), day(6)), registry)
        assert pos.tolist() == [2, 3]

    def test_same_range_of_another_identity_is_allowed(self):
        df = make_df(n=5)
        registry = self._registered(df, (day(0), day(2)), identity=("ETH", "1d"))
        pos, _ = self._resolve(df, (day(0), day(2)), registry)
        assert pos.tolist() == [0, 1, 2]

    def test_same_product_at_another_resolution_is_another_identity(self):
        df = make_df(n=5)
        registry = self._registered(df, (day(0), day(2)), identity=("BTC", "1h"))
        pos, _ = self._resolve(df, (day(0), day(2)), registry)
        assert pos.tolist() == [0, 1, 2]

    def test_empty_targets_never_overlap(self):
        df = make_df(n=5)
        registry = self._registered(df, (day(0), day(4)))
        pos, _ = self._resolve(df, (day(10), day(20)), registry)
        assert len(pos) == 0

    def test_registered_empty_segment_never_blocks(self):
        df = make_df(n=5)
        registry = self._registered(df, (day(10), day(20)))
        pos, _ = self._resolve(df, (day(0), day(4)), registry)
        assert pos.tolist() == [0, 1, 2, 3, 4]

    def test_overlap_is_checked_against_every_registered_entry(self):
        df = make_df(n=6)
        registry = {
            **self._registered(df, (day(0), day(1))),
            **self._registered(df, (day(4), day(5))),
        }
        assert self._resolve(df, (day(2), day(3)), registry)[0].tolist() == [2, 3]
        with pytest.raises(ValueError, match="a row can only be registered once"):
            self._resolve(df, (day(3), day(4)), registry)

    def test_key_identity_decides_which_entries_are_the_same_identity(self):
        # portfolio style: registry keyed (data_range, product), identity = product
        df = make_df(n=5)
        _, dates = self._resolve(df, (day(0), day(2)))
        registry = {((day(0), day(2)), "BTC"): {"dates": dates}}

        def by_product(key: tuple[Any, ...]) -> Any:
            return key[1]

        with pytest.raises(ValueError, match="BTC: data_range"):
            m._resolve_targets(registry, "BTC", by_product, df, (day(1), day(3)))
        pos, _ = m._resolve_targets(registry, "ETH", by_product, df, (day(1), day(3)))
        assert pos.tolist() == [1, 2, 3]
