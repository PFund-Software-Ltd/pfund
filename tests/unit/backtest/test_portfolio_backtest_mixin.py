"""Smoke tests for portfolio backtesting (FAST mode).

The golden case pins the exact numeric outputs against a fixture generated
BEFORE the registration-consistency fixes (data_range=None wipe semantics,
sorted-dates guard, duplicate-dates guard) — explicit-range results must
never change.
"""

import datetime
import pathlib

import numpy as np
import polars as pl
import pytest

from pfund._backtest.backtest_mixin import setup_backtest_df
from synthetic import build_df, run_golden_portfolio

GOLDEN_DIR = pathlib.Path(__file__).parents[2] / "data" / "golden"
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

FIRST_HALF = (datetime.date(2025, 1, 1), datetime.date(2025, 1, 6))
SECOND_HALF = (datetime.date(2025, 1, 7), datetime.date(2025, 1, 12))


def col(df, name: str) -> np.ndarray:
    import narwhals as nw

    return np.asarray(nw.from_native(df).get_column(name).to_numpy(), dtype=np.float64)


@pytest.fixture(params=["polars", "pandas"])
def backend(request):
    return request.param


@pytest.mark.smoke
def test_golden_portfolio(backend):
    golden = pl.read_parquet(GOLDEN_DIR / "portfolio_backtest.parquet")
    result = run_golden_portfolio(backend)
    for c in OUTPUT_COLS:
        np.testing.assert_array_equal(col(result, c), golden[c].to_numpy(), err_msg=c)


def test_data_range_none_wipes_ranges(backend):
    """create_weight(data_range=None) wipes the product's prior registrations
    (reconfigure + rerun) — the latest single-shot call wins."""
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    df.create_weight(0.5)
    df.create_weight(0.3)  # would raise "overlaps" without the wipe
    result = df.backtest()

    fresh = setup_backtest_df(build_df(backend, ["BTC"]))
    fresh.create_weight(0.3)
    expected = fresh.backtest()
    for c in OUTPUT_COLS:
        np.testing.assert_array_equal(col(result, c), col(expected, c), err_msg=c)


def test_none_wipes_explicit_ranges_too(backend):
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    df.create_weight(0.5, data_range=FIRST_HALF)
    df.create_weight(0.3)  # single-shot: wipes the explicit range, no overlap error
    result = df.backtest()
    # the wiped first-half instruction is gone: 0.3 lands at the LAST row only
    weight = col(result, "weight")
    assert np.isnan(weight[:-1]).all()
    assert weight[-1] == 0.3


def test_mismatched_dates_guard(backend):
    """Configuring on a row-subset df without data_range then backtesting the
    full df must raise (same alignment guard as product backtesting)."""
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    cutoff = datetime.datetime(2025, 1, 6)
    if backend == "polars":
        subset = df.filter(pl.col("date") <= cutoff)
    else:
        subset = df[df["date"] <= cutoff]
    subset.create_weight(0.5)
    with pytest.raises(ValueError, match="mismatched dates"):
        df.backtest()


def test_create_weight_requires_sorted_dates(backend):
    """A scalar weight anchors at the range's LAST row — positional, so an
    unsorted df must be rejected at registration."""
    df = build_df(backend, ["BTC"])
    setup_backtest_df(df)  # attaches the methods to the df class
    if backend == "polars":
        unsorted_df = df.reverse()
    else:
        unsorted_df = df.iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        unsorted_df.create_weight(0.5)


def test_duplicate_dates_rejected(backend):
    """One product at two resolutions duplicates dates — that would silently
    double-write panel rows at backtest(); rejected at registration."""
    df = build_df(backend, ["BTC"])
    if backend == "polars":
        two_res = pl.concat(
            [df, df.with_columns(pl.lit("2d").alias("resolution"))]
        ).sort("date")
    else:
        import pandas as pd

        two_res = (
            pd.concat([df, df.assign(resolution="2d")])
            .sort_values("date", kind="stable")
            .reset_index(drop=True)
        )
    setup_backtest_df(two_res)
    with pytest.raises(ValueError, match="duplicate dates"):
        two_res.create_weight(0.5)


def test_overlapping_ranges_raise(backend):
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    df.create_weight(0.5, data_range=FIRST_HALF)
    # partial overlap
    with pytest.raises(ValueError, match="overlaps"):
        df.create_weight(
            0.3, data_range=(datetime.date(2025, 1, 3), datetime.date(2025, 1, 8))
        )
    # identical range is an overlap too
    with pytest.raises(ValueError, match="overlaps"):
        df.create_weight(0.3, data_range=FIRST_HALF)


def test_empty_range_is_noop(backend):
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    df.create_weight(
        0.5, data_range=(datetime.date(2025, 2, 1), datetime.date(2025, 2, 5))
    )
    df.create_weight(0.5, data_range=FIRST_HALF)
    result = df.backtest()
    # the empty range registered nothing; the real range's scalar weight
    # lands at its last row (Jan 6 = row index 5)
    weight = col(result, "weight")
    assert weight[5] == 0.5
    assert np.isnan(np.delete(weight, 5)).all()
