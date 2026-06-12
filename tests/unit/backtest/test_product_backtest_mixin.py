"""Smoke tests for product backtesting (FAST mode).

Golden cases pin the exact numeric outputs of fixed-period single-/multi-
product backtesting against fixtures generated BEFORE dynamic time segments
(create_signal(data_range=...)) were added — they must never change.

The dynamic-segment layer scatters per-segment INSTRUCTIONS onto each combo's
full rows (registration controls instructions, never data): the kernel always
runs once per combo over all its bars, so positions carry across segments and
stops stay live on every bar — no force-close at segment ends.
"""

import datetime
import pathlib

import numpy as np
import polars as pl
import pytest

from pfund._backtest.backtest_mixin import setup_backtest_df
from synthetic import (
    BTC_SIGNAL,
    DATES,
    build_df,
    build_signal,
    run_golden_multi,
    run_golden_single,
)

GOLDEN_DIR = pathlib.Path(__file__).parents[2] / "data" / "golden"
OUTPUT_COLS = [
    "signal",
    "order_price",
    "order_size",
    "trade_price",
    "trade_size",
    "position",
    "avg_price",
]

# BTC closes (synthetic.py): [100, 102, 104, 103, 96, 99, 101, 103, 106, 108, 103, 104]
FIRST_HALF = (datetime.date(2025, 1, 1), datetime.date(2025, 1, 6))
SECOND_HALF = (datetime.date(2025, 1, 7), datetime.date(2025, 1, 12))


def col(df, name: str) -> np.ndarray:
    import narwhals as nw

    return np.asarray(nw.from_native(df).get_column(name).to_numpy(), dtype=np.float64)


def assert_outputs_equal(result, golden: pl.DataFrame, cols: list[str]) -> None:
    for c in cols:
        np.testing.assert_array_equal(col(result, c), golden[c].to_numpy(), err_msg=c)


@pytest.fixture(params=["polars", "pandas"])
def backend(request):
    return request.param


def _signal_at(backend, dates_with_buy: list[datetime.datetime]):
    return build_signal(backend, [1.0 if d in dates_with_buy else None for d in DATES])


def _all_buy(backend):
    return build_signal(backend, [1.0] * len(DATES))


# ====================================================================
# Golden cases — fixed period, results pinned by pre-refactor fixtures
# ====================================================================


@pytest.mark.smoke
def test_golden_single_product(backend):
    golden = pl.read_parquet(GOLDEN_DIR / "product_backtest_single.parquet")
    result = run_golden_single(backend)
    assert_outputs_equal(result, golden, OUTPUT_COLS + ["stop_price"])


@pytest.mark.smoke
def test_golden_multi_product(backend):
    golden = pl.read_parquet(GOLDEN_DIR / "product_backtest_multi.parquet")
    result = run_golden_multi(backend)
    assert_outputs_equal(result, golden, OUTPUT_COLS + ["stop_price"])


@pytest.mark.smoke
def test_segment_equivalence(backend):
    """Two segments with identical params covering the full span must equal
    the never-segmented golden run — scatter + extend-forward is lossless."""
    golden = pl.read_parquet(GOLDEN_DIR / "product_backtest_single.parquet")
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    for data_range in (FIRST_HALF, SECOND_HALF):
        (
            df.create_signal(
                signal=build_signal(backend, BTC_SIGNAL), data_range=data_range
            )
            .open_position(order_quantity=2)
            .close_position(stop_loss=0.05)
        )
    result = df.backtest()
    assert_outputs_equal(result, golden, OUTPUT_COLS + ["stop_price"])


# ====================================================================
# Dynamic time segments
# ====================================================================


def test_first_only_true_then_false(backend):
    """Segment 1 first_only=True trades once; segment 2 first_only=False
    re-enables trading on the SAME (unchanged) signal streak."""
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    for data_range, first_only in ((FIRST_HALF, True), (SECOND_HALF, False)):
        (
            df.create_signal(signal=_all_buy(backend), data_range=data_range)
            .open_position(order_quantity=1, first_only=first_only)
            .close_position()
        )
    result = df.backtest()
    trade_size = col(result, "trade_size")

    # segment 1: only the first order (bar 0) fills (bar 1); rest blocked
    assert trade_size[1] == 1.0
    assert np.isnan(trade_size[2:7]).all()  # incl. bar 6 (bar 5's order was blocked)
    # segment 2: trading re-enabled — orders at bars 6..10 fill at bars 7..11
    np.testing.assert_array_equal(trade_size[7:], np.ones(5))
    assert col(result, "position")[-1] == 6.0


def test_first_only_false_then_true(backend):
    """Segment 1 first_only=False trades every bar; segment 2 first_only=True
    blocks — the streak already traded and the signal never changes."""
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    for data_range, first_only in ((FIRST_HALF, False), (SECOND_HALF, True)):
        (
            df.create_signal(signal=_all_buy(backend), data_range=data_range)
            .open_position(order_quantity=1, first_only=first_only)
            .close_position()
        )
    result = df.backtest()
    trade_size = col(result, "trade_size")
    order_size = col(result, "order_size")

    # segment 1: orders at bars 0..5 fill at bars 1..6
    np.testing.assert_array_equal(trade_size[1:7], np.ones(6))
    # segment 2: no new orders (streak already traded), no fills after bar 6
    assert np.isnan(order_size[6:]).all()
    assert np.isnan(trade_size[7:]).all()
    assert col(result, "position")[-1] == 6.0


def test_per_segment_stop_change_mid_position(backend):
    """Segment 2 adds a take_profit to a position opened in segment 1: stops
    are re-placed every bar from the unchanged avg_price, so the new param
    applies to the carried position."""
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    for data_range, take_profit in ((FIRST_HALF, None), (SECOND_HALF, 0.05)):
        (
            df.create_signal(
                signal=_signal_at(backend, [DATES[0]]), data_range=data_range
            )
            .open_position(order_quantity=1)
            .close_position(take_profit=take_profit)
        )
    result = df.backtest()
    trade_price = col(result, "trade_price")
    trade_size = col(result, "trade_size")
    position = col(result, "position")

    # entry: buy 1 fills at bar 1 (prev close 100); no TP in segment 1 even
    # though bar 8's high (107) would have breached 105
    assert trade_size[1] == 1.0 and trade_price[1] == 100.0
    assert np.isnan(trade_size[2:8]).all()
    # segment 2's TP = 100 * 1.05 triggers at bar 8 (high 107 >= 105)
    assert trade_price[8] == pytest.approx(105.0) and trade_size[8] == -1.0
    assert position[8] == 0.0


def test_carry_through_uncovered_gap(backend):
    """A position carried through rows covered by no segment keeps drifting
    with the previous segment's stops live (extend-forward)."""
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    seg1 = (datetime.date(2025, 1, 1), datetime.date(2025, 1, 3))
    seg2 = (datetime.date(2025, 1, 9), datetime.date(2025, 1, 12))
    (
        df.create_signal(signal=_signal_at(backend, [DATES[0]]), data_range=seg1)
        .open_position(order_quantity=1)
        .close_position(stop_loss=0.05)
    )
    (
        df.create_signal(signal=_signal_at(backend, [DATES[8]]), data_range=seg2)
        .open_position(order_quantity=1)
        .close_position(stop_loss=0.05)
    )
    result = df.backtest()
    signal = col(result, "signal")
    trade_price = col(result, "trade_price")
    trade_size = col(result, "trade_size")
    position = col(result, "position")

    # gap rows (3..7) are uncovered: no instructions, but real position state
    assert np.isnan(signal[3:8]).all()
    assert not np.isnan(position[3:8]).any()
    # entry in seg1: buy 1 fills at bar 1 at 100 → stop at 95 (extends into the gap)
    assert trade_size[1] == 1.0 and trade_price[1] == 100.0
    assert position[3] == 1.0  # carried into the gap
    # seg1's stop triggers DURING the gap: bar 4's low (95) hits 95
    assert trade_price[4] == 95.0 and trade_size[4] == -1.0
    assert position[4] == 0.0
    # seg2 trades independently: buy at bar 8 fills at bar 9 (prev close 106)
    assert trade_size[9] == 1.0 and trade_price[9] == 106.0
    assert position[-1] == 1.0  # last standing position stays open


def test_dynamic_universe_multi_product(backend):
    """BTC instructed early, ETH instructed later: each product only receives
    instructions inside its own segment; positions carry afterwards (no
    force-close), and a combo's rows outside segments still have real state."""
    df = setup_backtest_df(build_df(backend, ["BTC", "ETH"]))
    if backend == "polars":

        def filter_product(product):
            return df.filter(pl.col("product") == product)
    else:

        def filter_product(product):
            return df[df["product"] == product]

    ranges = {
        "BTC": (datetime.date(2025, 1, 1), datetime.date(2025, 1, 3)),
        "ETH": (datetime.date(2025, 1, 4), datetime.date(2025, 1, 6)),
    }
    first_signal_date = {"BTC": DATES[0], "ETH": DATES[3]}
    for product in ["BTC", "ETH"]:
        (
            filter_product(product)
            .create_signal(
                signal=_signal_at(backend, [first_signal_date[product]]),
                data_range=ranges[product],
            )
            .open_position(order_quantity=1)
            .close_position()
        )
    result = df.backtest()

    import narwhals as nw

    products = nw.from_native(result).get_column("product").to_numpy()
    signal = col(result, "signal")
    trade_size = col(result, "trade_size")
    position = col(result, "position")

    btc, eth = products == "BTC", products == "ETH"
    # signals only inside each product's segment (3 covered rows each)
    assert np.nansum(np.abs(signal[btc])) == 1.0
    assert np.nansum(np.abs(signal[eth])) == 1.0
    # BTC: entry fills at its 2nd bar, position carried to the end (no force-close)
    assert np.nansum(trade_size[btc]) == 1.0
    assert position[btc][-1] == 1.0
    # ETH: flat (real 0, not nan) before its segment, then entry fills
    assert (position[eth][:4] == 0.0).all()
    assert np.nansum(trade_size[eth]) == 1.0
    assert position[eth][-1] == 1.0


# ====================================================================
# Registration rules
# ====================================================================


def test_overlapping_ranges_raise(backend):
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    df.create_signal(signal=_signal_at(backend, [DATES[0]]), data_range=FIRST_HALF)
    # partial overlap
    with pytest.raises(ValueError, match="overlaps"):
        df.create_signal(
            signal=_signal_at(backend, [DATES[2]]),
            data_range=(datetime.date(2025, 1, 3), datetime.date(2025, 1, 8)),
        )
    # identical range is an overlap too — a row's instruction is registered once
    with pytest.raises(ValueError, match="overlaps"):
        df.create_signal(signal=_signal_at(backend, [DATES[0]]), data_range=FIRST_HALF)


def test_data_range_none_wipes_segments(backend):
    """create_signal(data_range=None) wipes prior segments → results identical
    to the never-segmented golden single-product case."""
    golden = pl.read_parquet(GOLDEN_DIR / "product_backtest_single.parquet")
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    df.create_signal(signal=_signal_at(backend, [DATES[0]]), data_range=FIRST_HALF)
    (
        df.create_signal(signal=build_signal(backend, BTC_SIGNAL))
        .open_position(order_quantity=2)
        .close_position(stop_loss=0.05)
    )
    result = df.backtest()
    assert_outputs_equal(result, golden, OUTPUT_COLS + ["stop_price"])


def test_create_signal_requires_sorted_dates(backend):
    """seg_rows[0] anchors the extend-forward rule and signal ffill is
    positional — an unsorted df must be rejected at registration."""
    df = build_df(backend, ["BTC"])
    setup_backtest_df(df)  # attaches the methods to the df class
    if backend == "polars":
        unsorted_df = df.reverse()
    else:
        unsorted_df = df.iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        unsorted_df.create_signal(signal=_signal_at(backend, [DATES[0]]))


def test_open_position_requires_create_signal(backend):
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    with pytest.raises(ValueError, match="create_signal"):
        df.open_position(order_quantity=1)


def test_empty_range_is_noop(backend):
    """A range selecting no rows (product not listed yet) chains harmlessly;
    backtest() runs the other segments."""
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    (
        df.create_signal(
            signal=_signal_at(backend, []),
            data_range=(datetime.date(2025, 2, 1), datetime.date(2025, 2, 5)),
        )
        .open_position(order_quantity=1)
        .close_position()
    )
    (
        df.create_signal(signal=_signal_at(backend, [DATES[0]]), data_range=FIRST_HALF)
        .open_position(order_quantity=2)
        .close_position()
    )
    result = df.backtest()
    assert col(result, "trade_size")[1] == 2.0


def test_mismatched_dates_guard_preserved(backend):
    """Configuring on a row-subset df without data_range then backtesting the
    full df must still raise (golden alignment guard)."""
    df = setup_backtest_df(build_df(backend, ["BTC"]))
    cutoff = datetime.datetime(2025, 1, 6)
    if backend == "polars":
        subset = df.filter(pl.col("date") <= cutoff)
    else:
        subset = df[df["date"] <= cutoff]
    (
        subset.create_signal(signal=build_signal(backend, BTC_SIGNAL[:6]))
        .open_position(order_quantity=2)
        .close_position(stop_loss=0.05)
    )
    with pytest.raises(ValueError, match="mismatched dates"):
        df.backtest()
