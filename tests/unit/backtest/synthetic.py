"""Deterministic synthetic OHLCV data shared by the golden-fixture generator
(tests/data/golden/generate.py) and the product backtest smoke tests.

The price paths are crafted so the golden cases exercise real kernel paths:
- BTC with close_position(stop_loss=0.05): the day-5 dip triggers the SL
- ETH with close_position(take_profit=0.04): the day-4 high triggers the TP
"""

import datetime

DATES = [datetime.datetime(2025, 1, d) for d in range(1, 13)]

BTC_CLOSE = [
    100.0,
    102.0,
    104.0,
    103.0,
    96.0,
    99.0,
    101.0,
    103.0,
    106.0,
    108.0,
    103.0,
    104.0,
]
ETH_CLOSE = [
    50.0,
    51.0,
    52.0,
    53.0,
    51.0,
    50.0,
    52.0,
    54.0,
    55.0,
    53.0,
    52.0,
    53.0,
]
CLOSES = {"BTC": BTC_CLOSE, "ETH": ETH_CLOSE}

BTC_SIGNAL = [1.0, None, None, None, None, None, -1.0, None, None, 1.0, None, None]
ETH_SIGNAL = [None, 1.0, None, None, -1.0, None, None, 1.0, None, None, None, None]
SIGNALS = {"BTC": BTC_SIGNAL, "ETH": ETH_SIGNAL}


def _product_rows(product: str, closes: list[float]) -> list[dict]:
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        rows.append(
            {
                "date": DATES[i],
                "resolution": "1d",
                "product": product,
                "open": o,
                "high": max(o, c) + 1.0,
                "low": min(o, c) - 1.0,
                "close": c,
                "volume": 1000.0,
            }
        )
    return rows


def build_df(backend: str, products: list[str]):
    """Build a LONG-form OHLCV df sorted by (date, product)."""
    rows = []
    for product in products:
        rows.extend(_product_rows(product, CLOSES[product]))
    rows.sort(key=lambda r: (r["date"], r["product"]))
    if backend == "polars":
        import polars as pl

        return pl.DataFrame(rows)
    elif backend == "pandas":
        import pandas as pd

        return pd.DataFrame(rows)
    raise ValueError(backend)


def build_signal(backend: str, values: list[float | None]):
    if backend == "polars":
        import polars as pl

        return pl.Series("signal", values, dtype=pl.Float64)
    elif backend == "pandas":
        import pandas as pd

        return pd.Series(values, dtype="float64")
    raise ValueError(backend)


def run_golden_single(backend: str):
    """Golden case 1: single product, fixed period (no data_range)."""
    from pfund._backtest.backtest_mixin import setup_backtest_df

    df = build_df(backend, ["BTC"])
    df = setup_backtest_df(df)
    (
        df.create_signal(signal=build_signal(backend, BTC_SIGNAL))
        .open_position(order_quantity=2)
        .close_position(stop_loss=0.05)
    )
    return df.backtest()


def run_golden_portfolio(backend: str):
    """Golden portfolio case: two products, two periods, scalar + series weights."""
    import datetime as dt

    from pfund._backtest.backtest_mixin import setup_backtest_df

    df = setup_backtest_df(build_df(backend, ["BTC", "ETH"]))
    first_half = (dt.date(2025, 1, 1), dt.date(2025, 1, 6))
    second_half = (dt.date(2025, 1, 7), dt.date(2025, 1, 12))
    if backend == "polars":
        import polars as pl

        def filter_product(product):
            return df.filter(pl.col("product") == product)

        def weight_series(values):
            return pl.Series("weight", values, dtype=pl.Float64)
    else:
        import pandas as pd

        def filter_product(product):
            return df[df["product"] == product]

        def weight_series(values):
            return pd.Series(values, dtype="float64")

    btc, eth = filter_product("BTC"), filter_product("ETH")
    # scalar weights: one rebalance instruction at the range's last row
    btc.create_weight(0.6, data_range=first_half)
    btc.create_weight(0.4, data_range=second_half)
    eth.create_weight(0.4, data_range=first_half)
    # series weights: positional over the range's 6 rows (rebalance, drift, close)
    eth.create_weight(
        weight_series([None, 0.2, None, None, None, 0.0]), data_range=second_half
    )
    return df.backtest(initial_capital=1_000_000, compound=True)


def run_golden_multi(backend: str):
    """Golden case 2: two products, fixed period, per-combo configure loop."""
    from pfund._backtest.backtest_mixin import setup_backtest_df

    df = build_df(backend, ["BTC", "ETH"])
    df = setup_backtest_df(df)
    close_kwargs = {
        "BTC": {"stop_loss": 0.05},
        "ETH": {"take_profit": 0.04},
    }
    if backend == "polars":
        import polars as pl

        def filter_product(product):
            return df.filter(pl.col("product") == product)
    else:

        def filter_product(product):
            return df[df["product"] == product]

    for product in ["BTC", "ETH"]:
        product_df = filter_product(product)
        signal = build_signal(backend, SIGNALS[product])
        (
            product_df.create_signal(signal=signal)
            .open_position(order_quantity=2)
            .close_position(**close_kwargs[product])
        )
    return df.backtest()
