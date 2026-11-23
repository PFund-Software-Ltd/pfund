"""Tests _split_df_into_time_chunks(), which splits the backtest df into contiguous date ranges for chunked backtesting."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from pfund.engines.backtest_engine import _split_df_into_time_chunks


def _dates(n: int) -> list[datetime.datetime]:
    start = datetime.datetime(2025, 1, 1)
    return [start + datetime.timedelta(minutes=i) for i in range(n)]


def _wide_df(num_dates: int) -> pl.DataFrame:
    """One row per date."""
    return pl.DataFrame({"date": _dates(num_dates), "x": list(range(num_dates))})


def _long_df(num_dates: int, products: list[str]) -> pl.DataFrame:
    """One row per (date, product), sorted by date first like KEY_COLS."""
    rows = [
        {"date": date, "product": product, "x": i}
        for i, date in enumerate(_dates(num_dates))
        for product in products
    ]
    return pl.DataFrame(rows)


def _chunk_dates(chunk: pl.DataFrame) -> list[datetime.datetime]:
    return chunk.get_column("date").unique(maintain_order=True).to_list()


class TestSplitDfIntoTimeChunks:
    def test_single_chunk_is_the_whole_df(self):
        df = _wide_df(5)
        chunks = _split_df_into_time_chunks(df, 1)
        assert len(chunks) == 1
        chunk, chunk_num = chunks[0]
        assert chunk_num == 0
        assert chunk.equals(df)

    def test_chunk_nums_are_in_order(self):
        chunks = _split_df_into_time_chunks(_wide_df(10), 4)
        assert [chunk_num for _, chunk_num in chunks] == [0, 1, 2, 3]

    def test_chunks_are_contiguous_and_cover_every_row(self):
        df = _wide_df(10)
        chunks = _split_df_into_time_chunks(df, 3)
        recombined = pl.concat([chunk for chunk, _ in chunks], how="vertical")
        assert recombined.equals(df)

    def test_chunk_sizes_follow_array_split(self):
        # 10 dates into 3 chunks -> 4, 3, 3 dates (np.array_split puts the extra dates first)
        chunks = _split_df_into_time_chunks(_wide_df(10), 3)
        assert [len(chunk) for chunk, _ in chunks] == [4, 3, 3]

    def test_long_df_never_splits_a_timestamp(self):
        products = ["BTC", "ETH", "SOL"]
        df = _long_df(7, products)
        chunks = _split_df_into_time_chunks(df, 3)

        seen: set[datetime.datetime] = set()
        for chunk, _ in chunks:
            dates = _chunk_dates(chunk)
            # no date appears in more than one chunk
            assert seen.isdisjoint(dates)
            seen.update(dates)
            # every date in a chunk has all products, i.e. the timestamp was not cut
            assert len(chunk) == len(dates) * len(products)
        assert len(seen) == 7

    def test_chunks_split_by_unique_dates_not_rows(self):
        # 4 dates x 2 products = 8 rows, 2 chunks -> 2 dates (4 rows) each
        chunks = _split_df_into_time_chunks(_long_df(4, ["BTC", "ETH"]), 2)
        assert [len(_chunk_dates(chunk)) for chunk, _ in chunks] == [2, 2]
        assert [len(chunk) for chunk, _ in chunks] == [4, 4]

    def test_num_chunks_equal_to_num_dates(self):
        chunks = _split_df_into_time_chunks(_long_df(3, ["BTC", "ETH"]), 3)
        assert [len(_chunk_dates(chunk)) for chunk, _ in chunks] == [1, 1, 1]

    def test_raises_when_num_chunks_exceeds_num_dates(self):
        # 6 rows but only 3 unique dates
        with pytest.raises(ValueError, match="cannot exceed the number of unique"):
            _split_df_into_time_chunks(_long_df(3, ["BTC", "ETH"]), 4)

    def test_raises_when_num_chunks_below_one(self):
        with pytest.raises(ValueError, match="greater than 0"):
            _split_df_into_time_chunks(_wide_df(3), 0)

    def test_raises_without_date_column(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="requires a 'date' column"):
            _split_df_into_time_chunks(df, 2)

    def test_raises_on_empty_df(self):
        df = pl.DataFrame({"date": []}, schema={"date": pl.Datetime("us")})
        with pytest.raises(ValueError, match="at least one row"):
            _split_df_into_time_chunks(df, 1)

    def test_raises_on_null_dates(self):
        dates: list[datetime.datetime | None] = [*_dates(2), None]
        df = pl.DataFrame({"date": dates})
        with pytest.raises(ValueError, match="null dates"):
            _split_df_into_time_chunks(df, 2)

    def test_raises_when_not_sorted_by_date(self):
        df = pl.DataFrame({"date": list(reversed(_dates(3)))})
        with pytest.raises(ValueError, match="sorted by 'date'"):
            _split_df_into_time_chunks(df, 2)
