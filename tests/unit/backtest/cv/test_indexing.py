from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pfund.backtest.cv.indexing import (
    expand_time_indices,
    normalize_indices,
    resolve_time_spine,
    row_count,
    take_rows,
    validate_aligned,
)

N_ROWS = 5

# each factory builds N_ROWS rows of the same values in a different container
DATA_FACTORIES: dict[str, Callable[[], Any]] = {
    "numpy_1d": lambda: np.arange(N_ROWS) * 10,
    "numpy_2d": lambda: np.arange(N_ROWS * 2).reshape(N_ROWS, 2),
    "polars_df": lambda: pl.DataFrame({"a": range(N_ROWS), "b": range(N_ROWS)}),
    "polars_series": lambda: pl.Series("a", range(N_ROWS)),
    "pandas_df": lambda: pd.DataFrame({"a": range(N_ROWS), "b": range(N_ROWS)}),
    "pandas_series": lambda: pd.Series(range(N_ROWS), name="a"),
}


@pytest.fixture(params=list(DATA_FACTORIES), ids=list(DATA_FACTORIES))
def data(request: pytest.FixtureRequest) -> Any:
    return DATA_FACTORIES[request.param]()


def _first_column(data: Any) -> list[Any]:
    """Values of the first column (or the series/1d array itself) as a list."""
    if isinstance(data, np.ndarray):
        return (data if data.ndim == 1 else data[:, 0]).tolist()
    if isinstance(data, (pl.DataFrame, pd.DataFrame)):
        return list(data["a"])
    return list(data)


class TestRowCount:
    def test_counts_rows(self, data: Any):
        assert row_count(data) == N_ROWS

    def test_2d_array_counts_first_axis(self):
        assert row_count(np.zeros((3, 7))) == 3

    def test_empty_dataframe(self):
        assert row_count(pl.DataFrame({"a": []})) == 0

    def test_0d_array_raises(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            row_count(np.array(1))


class TestNormalizeIndices:
    def test_returns_int64(self):
        result = normalize_indices(
            np.array([2, 0], dtype=np.int32), name="idx", n_rows=3
        )
        assert result.dtype == np.int64
        assert result.tolist() == [2, 0]

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.uint16, np.uint64])
    def test_accepts_any_integer_dtype(self, dtype: type[np.integer[Any]]):
        result = normalize_indices(np.array([0, 1], dtype=dtype), name="idx", n_rows=2)
        assert result.dtype == np.int64

    def test_preserves_order(self):
        result = normalize_indices(np.array([3, 1, 2]), name="idx", n_rows=4)
        assert result.tolist() == [3, 1, 2]

    def test_result_is_read_only(self):
        result = normalize_indices(np.array([0, 1]), name="idx", n_rows=2)
        assert not result.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            result[0] = 1

    def test_copies_input(self):
        """The caller mutating its array afterwards must not change the selectors."""
        indices = np.array([0, 1], dtype=np.int64)
        result = normalize_indices(indices, name="idx", n_rows=2)
        assert not np.shares_memory(result, indices)
        indices[0] = 1
        assert result.tolist() == [0, 1]
        # the caller's array is left writable
        assert indices.flags.writeable

    def test_boundaries_are_valid(self):
        result = normalize_indices(np.array([0, 4]), name="idx", n_rows=5)
        assert result.tolist() == [0, 4]

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="idx must be one-dimensional"):
            normalize_indices(np.array([[0, 1]]), name="idx", n_rows=2)

    def test_0d_raises(self):
        with pytest.raises(ValueError, match="idx must be one-dimensional"):
            normalize_indices(np.array(0), name="idx", n_rows=2)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="idx must not be empty"):
            normalize_indices(np.array([], dtype=np.int64), name="idx", n_rows=2)

    @pytest.mark.parametrize(
        "indices",
        [np.array([0.0, 1.0]), np.array([True, False]), np.array(["0", "1"])],
        ids=["float", "bool", "str"],
    )
    def test_non_integer_dtype_raises(self, indices: np.ndarray[Any, Any]):
        with pytest.raises(TypeError, match="idx must contain integer row positions"):
            normalize_indices(indices, name="idx", n_rows=2)

    @pytest.mark.parametrize("bad", [-1, 5, 100], ids=["negative", "n_rows", "far"])
    def test_out_of_bounds_raises(self, bad: int):
        with pytest.raises(IndexError, match="outside a dataset of 5 rows"):
            normalize_indices(np.array([0, bad]), name="idx", n_rows=5)

    def test_duplicates_raise(self):
        with pytest.raises(ValueError, match="idx must not contain duplicate"):
            normalize_indices(np.array([1, 0, 1]), name="idx", n_rows=3)

    def test_name_appears_in_message(self):
        with pytest.raises(ValueError, match=r"^train_indices "):
            normalize_indices(
                np.array([], dtype=np.int64), name="train_indices", n_rows=2
            )


class TestValidateAligned:
    def test_aligned_passes(self, data: Any):
        reference = pl.DataFrame({"x": range(N_ROWS)})
        validate_aligned(data, reference)

    @pytest.mark.parametrize("n_rows", [N_ROWS - 1, N_ROWS + 1])
    def test_misaligned_raises(self, n_rows: int):
        reference = pl.DataFrame({"x": range(N_ROWS)})
        with pytest.raises(
            ValueError, match=rf"\({N_ROWS} rows\), but got {n_rows} rows"
        ):
            validate_aligned(np.arange(n_rows), reference)


class TestTakeRows:
    def test_keeps_input_type(self, data: Any):
        result = take_rows(data, np.array([0, 2], dtype=np.int64))
        assert type(result) is type(data)
        assert row_count(result) == 2

    def test_selects_rows_in_given_order(self, data: Any):
        result = take_rows(data, np.array([3, 0, 4], dtype=np.int64))
        expected = [_first_column(data)[i] for i in (3, 0, 4)]
        assert _first_column(result) == expected

    def test_keeps_all_columns(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = take_rows(df, np.array([2, 1], dtype=np.int64))
        assert result.to_dict(as_series=False) == {"a": [3, 2], "b": ["z", "y"]}

    def test_2d_array_keeps_columns(self):
        arr = np.arange(6).reshape(3, 2)
        result = take_rows(arr, np.array([2, 0], dtype=np.int64))
        assert result.tolist() == [[4, 5], [0, 1]]

    def test_does_not_mutate_input(self, data: Any):
        before = _first_column(data)
        take_rows(data, np.array([1], dtype=np.int64))
        assert _first_column(data) == before


class TestResolveTimeSpine:
    def test_unique_dates_and_row_positions(self):
        X = pl.DataFrame({"date": [1, 1, 2, 3, 3, 3], "v": range(6)})
        time_spine, row_time_indices = resolve_time_spine(X, operation="op")
        assert time_spine.tolist() == [1, 2, 3]
        assert row_time_indices.tolist() == [0, 0, 1, 2, 2, 2]
        assert row_time_indices.dtype == np.int64

    def test_one_row_per_date(self):
        X = pl.DataFrame({"date": [10, 20, 30]})
        time_spine, row_time_indices = resolve_time_spine(X, operation="op")
        assert time_spine.tolist() == [10, 20, 30]
        assert row_time_indices.tolist() == [0, 1, 2]

    def test_datetime_dates(self):
        dates = pl.datetime_range(
            datetime.datetime(2024, 1, 1),
            datetime.datetime(2024, 1, 3),
            interval="1d",
            eager=True,
        )
        X = pl.DataFrame({"date": pl.concat([dates, dates]).sort()})
        time_spine, row_time_indices = resolve_time_spine(X, operation="op")
        assert len(time_spine) == 3
        assert row_time_indices.tolist() == [0, 0, 1, 1, 2, 2]

    def test_pandas_input(self):
        X = pd.DataFrame({"date": [1, 1, 2]})
        time_spine, row_time_indices = resolve_time_spine(X, operation="op")
        assert time_spine.tolist() == [1, 2]
        assert row_time_indices.tolist() == [0, 0, 1]

    def test_missing_date_column_raises(self):
        with pytest.raises(ValueError, match="my op requires a 'date' column"):
            resolve_time_spine(pl.DataFrame({"x": [1]}), operation="my op")

    def test_empty_raises(self):
        X = pl.DataFrame({"date": []}, schema={"date": pl.Int64})
        with pytest.raises(ValueError, match="my op requires at least one row"):
            resolve_time_spine(X, operation="my op")

    def test_null_dates_raise(self):
        X = pl.DataFrame({"date": [1, None, 2]})
        with pytest.raises(ValueError, match="my op does not support null dates"):
            resolve_time_spine(X, operation="my op")

    def test_unsorted_raises(self):
        X = pl.DataFrame({"date": [2, 1, 3]})
        with pytest.raises(ValueError, match="sorted by 'date' before my op"):
            resolve_time_spine(X, operation="my op")


class TestExpandTimeIndices:
    # 3 timestamps over 6 rows: t0 -> rows 0,1; t1 -> row 2; t2 -> rows 3,4,5
    ROW_TIME_INDICES = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)

    def _expand(self, time_indices: Any) -> np.ndarray[Any, Any]:
        return expand_time_indices(
            np.asarray(time_indices),
            row_time_indices=self.ROW_TIME_INDICES,
            num_times=3,
            name="idx",
        )

    @pytest.mark.parametrize(
        ("time_indices", "rows"),
        [([0], [0, 1]), ([1], [2]), ([2], [3, 4, 5]), ([0, 2], [0, 1, 3, 4, 5])],
    )
    def test_expands_to_every_row_of_each_timestamp(
        self, time_indices: list[int], rows: list[int]
    ):
        assert self._expand(time_indices).tolist() == rows

    def test_result_is_sorted_by_row_position(self):
        assert self._expand([2, 0]).tolist() == [0, 1, 3, 4, 5]

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="idx must be one-dimensional"):
            self._expand([[0, 1]])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="idx must not be empty"):
            self._expand(np.array([], dtype=np.int64))

    def test_non_integer_raises(self):
        with pytest.raises(TypeError, match="idx must contain integer positions"):
            self._expand([0.0, 1.0])

    @pytest.mark.parametrize("bad", [-1, 3])
    def test_out_of_bounds_raises(self, bad: int):
        with pytest.raises(IndexError, match="outside a spine of 3 timestamps"):
            self._expand([0, bad])

    def test_duplicates_raise(self):
        with pytest.raises(ValueError, match="idx must not contain duplicate"):
            self._expand([1, 1])
