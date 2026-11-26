from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


import numpy as np
import pandas as pd
import polars as pl
import pytest

from pfund.backtest.cv.indexing import (
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
