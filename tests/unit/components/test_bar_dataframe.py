"""Tests pfund.components.bar_dataframe: long/wide reshaping and bar-spine alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import narwhals as nw
import polars as pl
import pytest

if TYPE_CHECKING:
    from pfund.components import bar_dataframe as bdf
else:
    # TODO: replace with `from pfund.components import bar_dataframe as bdf`
    # once `pfund.components` is importable again. Its `__init__` currently
    # fails during the refactor, so the module is loaded straight from its file.
    _path = (
        Path(__file__).parents[3] / "src" / "pfund" / "components" / "bar_dataframe.py"
    )
    _spec = importlib.util.spec_from_file_location("_bar_dataframe_under_test", _path)
    assert _spec is not None and _spec.loader is not None
    bdf = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = bdf
    _spec.loader.exec_module(bdf)

R = "1_MINUTE"
KEYS = ["date", "product", "resolution"]


def t(hour: int, minute: int, second: int = 0, microsecond: int = 0) -> datetime:
    return datetime(2026, 1, 1, hour, minute, second, microsecond)


def frame(data: dict[str, list[Any]], **kwargs: Any) -> nw.DataFrame[Any]:
    return nw.from_native(pl.DataFrame(data, **kwargs), eager_only=True)


def rows(df: nw.DataFrame[Any]) -> list[tuple[Any, ...]]:
    return df.to_native().rows()


@pytest.fixture
def spine() -> nw.DataFrame[Any]:
    """1-minute bars: BTC has no 09:02 bar (a gap), ETH stops after 09:01."""
    return frame(
        {
            "date": [t(9, 0), t(9, 1), t(9, 3), t(9, 0), t(9, 1)],
            "product": ["BTC", "BTC", "BTC", "ETH", "ETH"],
            "resolution": [R] * 5,
            "close": [100.0, 101.0, 103.0, 50.0, 51.0],
        }
    )


def events(
    times: list[datetime],
    products: list[str | None] | None = None,
    *,
    time_col: str = "news_date",
) -> nw.DataFrame[Any]:
    data: dict[str, list[Any]] = {time_col: times}
    overrides: dict[str, Any] = {}
    if products is not None:
        data["product"] = products
        overrides["product"] = pl.String
    data["headline"] = [f"h{i}" for i in range(len(times))]
    return frame(data, schema_overrides=overrides)


class TestReorderKeyCols:
    def test_long_moves_all_key_cols_left(self):
        df = frame(
            {
                "close": [1.0],
                "resolution": [R],
                "open": [2.0],
                "product": ["BTC"],
                "date": [t(9, 0)],
            }
        )

        assert bdf.reorder_key_cols(df).columns == [*KEYS, "close", "open"]

    def test_wide_moves_only_the_index_col_left(self):
        df = frame(
            {
                "BTC:1_MINUTE:close": [1.0],
                "date": [t(9, 0)],
                "ETH:1_MINUTE:close": [2.0],
            }
        )

        assert bdf.reorder_key_cols(df, df_form="wide").columns == [
            "date",
            "BTC:1_MINUTE:close",
            "ETH:1_MINUTE:close",
        ]

    def test_unsupported_form_raises(self):
        df = frame({"date": [t(9, 0)], "product": ["BTC"], "resolution": [R]})

        with pytest.raises(ValueError, match="unsupported dataframe form"):
            bdf.reorder_key_cols(df, df_form="tall")  # ty: ignore[invalid-argument-type]


class TestPivotLongToWide:
    def pivot(self, df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        return bdf.pivot_long_to_wide(
            df, index_col="date", pivot_cols=["product", "resolution"]
        )

    def test_multiple_value_cols_are_named_key_then_field(self):
        df = frame(
            {
                "date": [t(9, 1), t(9, 0), t(9, 0), t(9, 1)],
                "product": ["BTC", "BTC", "ETH", "ETH"],
                "resolution": [R] * 4,
                "close": [101.0, 100.0, 50.0, 51.0],
                "volume": [11, 10, 5, 6],
            }
        )

        wide = self.pivot(df)

        assert wide.to_native().rows(named=True) == [
            {
                "date": t(9, 0),
                "BTC:1_MINUTE:close": 100.0,
                "ETH:1_MINUTE:close": 50.0,
                "BTC:1_MINUTE:volume": 10,
                "ETH:1_MINUTE:volume": 5,
            },
            {
                "date": t(9, 1),
                "BTC:1_MINUTE:close": 101.0,
                "ETH:1_MINUTE:close": 51.0,
                "BTC:1_MINUTE:volume": 11,
                "ETH:1_MINUTE:volume": 6,
            },
        ]

    def test_single_value_col_still_gets_the_field_suffix(
        self, spine: nw.DataFrame[Any]
    ):
        wide = self.pivot(spine)

        assert wide.columns == ["date", "BTC:1_MINUTE:close", "ETH:1_MINUTE:close"]

    def test_missing_combination_becomes_null(self, spine: nw.DataFrame[Any]):
        wide = self.pivot(spine)

        assert rows(wide) == [
            (t(9, 0), 100.0, 50.0),
            (t(9, 1), 101.0, 51.0),
            (t(9, 3), 103.0, None),
        ]

    def test_pivot_values_are_percent_escaped(self):
        df = frame(
            {
                "date": [1, 1, 1],
                "product": ["a:b", "50%", "nul\x00"],
                "resolution": ["x"] * 3,
                "v": [1, 2, 3],
            }
        )

        wide = self.pivot(df)

        assert set(wide.columns) == {"date", "a%3Ab:x:v", "50%25:x:v", "nul%00:x:v"}

    def test_value_col_named_like_the_internal_pivot_key_survives(self):
        df = frame(
            {"date": [1], "product": ["BTC"], "resolution": [R], "__pivot_key__": [7]}
        )

        wide = self.pivot(df)

        assert wide.to_native().rows(named=True) == [
            {"date": 1, "BTC:1_MINUTE:__pivot_key__": 7}
        ]

    def test_null_key_raises(self):
        df = frame(
            {
                "date": [1, 2],
                "product": ["BTC", None],
                "resolution": [R, R],
                "v": [1, 2],
            }
        )

        with pytest.raises(
            ValueError, match=r"cannot contain null values: \['product'\]"
        ):
            self.pivot(df)

    def test_non_string_pivot_col_raises(self):
        df = frame({"date": [1], "product": [1], "resolution": [R], "v": [1]})

        with pytest.raises(TypeError, match=r"non-string columns: \['product'\]"):
            self.pivot(df)

    def test_duplicate_key_raises(self):
        df = frame(
            {
                "date": [1, 1],
                "product": ["BTC", "BTC"],
                "resolution": [R, R],
                "v": [1, 2],
            }
        )

        with pytest.raises(ValueError, match="must form a unique composite key"):
            self.pivot(df)


class TestUnpivotWideToLong:
    def unpivot(self, df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        return bdf.unpivot_wide_to_long(
            df, index_col="date", pivot_cols=["product", "resolution"]
        )

    def round_trip(self, df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        wide = bdf.pivot_long_to_wide(
            df, index_col="date", pivot_cols=["product", "resolution"]
        )
        return self.unpivot(wide)

    def test_round_trip_restores_the_long_frame(self):
        df = frame(
            {
                "date": [t(9, 0), t(9, 0), t(9, 1), t(9, 1)],
                "product": ["BTC", "ETH", "BTC", "ETH"],
                "resolution": [R] * 4,
                "close": [100.0, 50.0, 101.0, 51.0],
                "volume": [10, 5, 11, 6],
            }
        )

        assert self.round_trip(df).to_native().equals(df.to_native())

    @pytest.mark.parametrize(
        "product",
        ["a:b", "50%", "%3A", "%25", "%2525", "nul\x00", ":%:", "", "under_score"],
    )
    def test_round_trip_restores_awkward_identifiers(self, product: str):
        df = frame(
            {
                "date": [1, 1],
                "product": [product, "plain"],
                "resolution": ["r:1", "r:1"],
                "v": [1, 2],
            }
        )

        long = self.round_trip(df)

        assert sorted(rows(long)) == sorted(rows(df))

    def test_combination_absent_from_the_long_frame_is_not_resurrected(
        self, spine: nw.DataFrame[Any]
    ):
        assert self.round_trip(spine).to_native().equals(spine.sort(KEYS).to_native())

    def test_row_is_kept_when_any_value_field_is_non_null(self):
        df = frame(
            {
                "date": [1, 2],
                "product": ["BTC", "BTC"],
                "resolution": [R, R],
                "a": [None, None],
                "b": [None, 5],
            },
            schema_overrides={"a": pl.Int64, "b": pl.Int64},
        )

        assert rows(self.round_trip(df)) == [(2, "BTC", R, None, 5)]

    def test_field_name_may_contain_colons(self):
        wide = frame({"date": [1], "BTC:1_MINUTE:ratio:a:b": [0.5]})

        long = self.unpivot(wide)

        assert long.to_native().rows(named=True) == [
            {"date": 1, "product": "BTC", "resolution": R, "ratio:a:b": 0.5}
        ]

    def test_output_is_sorted_by_index_then_pivot_cols(self):
        wide = frame(
            {"date": [2, 1], "ETH:1_MINUTE:v": [4, 3], "BTC:1_MINUTE:v": [2, 1]}
        )

        assert rows(self.unpivot(wide)) == [
            (1, "BTC", R, 1),
            (1, "ETH", R, 3),
            (2, "BTC", R, 2),
            (2, "ETH", R, 4),
        ]

    def test_empty_pivot_cols_raises(self):
        wide = frame({"date": [1], "BTC:1_MINUTE:v": [1]})

        with pytest.raises(ValueError, match="at least one column"):
            bdf.unpivot_wide_to_long(wide, index_col="date", pivot_cols=[])

    def test_missing_index_col_raises(self):
        wide = frame({"BTC:1_MINUTE:v": [1]})

        with pytest.raises(ValueError, match="index column 'date' not found"):
            self.unpivot(wide)

    def test_no_value_columns_raises(self):
        with pytest.raises(ValueError, match="no pivoted value columns"):
            self.unpivot(frame({"date": [1]}))

    def test_column_with_too_few_parts_raises(self):
        wide = frame({"date": [1], "BTC:close": [1.0]})

        with pytest.raises(
            ValueError, match="'BTC:close' is not a valid pivoted column"
        ):
            self.unpivot(wide)

    def test_two_columns_decoding_to_the_same_field_raises(self):
        wide = frame({"date": [1], "%:x:v": [1], "%25:x:v": [2]})

        with pytest.raises(ValueError, match="duplicate value field 'v'"):
            self.unpivot(wide)

    def test_pivot_key_missing_a_value_field_raises(self):
        wide = frame({"date": [1], "BTC:x:v": [1], "BTC:x:w": [2], "ETH:x:v": [3]})

        with pytest.raises(ValueError, match=r"missing value fields: \['w'\]"):
            self.unpivot(wide)


class TestAddBarCloseColumn:
    def test_close_is_one_millisecond_before_the_next_open(self):
        df = frame({"date": [t(9, 0), t(9, 1)], "resolution": [R, R]})

        result = bdf.add_bar_close_column(df)

        assert result.columns == ["date", "resolution", "_bar_close"]
        assert result.get_column("_bar_close").to_list() == [
            t(9, 0, 59, 999_000),
            t(9, 1, 59, 999_000),
        ]

    def test_custom_column_names(self):
        df = frame({"ts": [t(9, 0)], "res": ["1_DAY"]})

        result = bdf.add_bar_close_column(
            df, date_col="ts", resolution_col="res", output_col="available_at"
        )

        assert result.get_column("available_at").to_list() == [
            t(9, 0) + timedelta(days=1, milliseconds=-1)
        ]

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match=r"missing columns \['resolution'\]"):
            bdf.add_bar_close_column(frame({"date": [t(9, 0)]}))

    def test_null_resolution_raises(self):
        df = frame({"date": [t(9, 0), t(9, 1)], "resolution": [R, None]})

        with pytest.raises(ValueError, match="'resolution' cannot contain null values"):
            bdf.add_bar_close_column(df)

    def test_multiple_resolutions_raise(self):
        df = frame({"date": [t(9, 0), t(9, 0)], "resolution": [R, "2_MINUTE"]})

        with pytest.raises(ValueError, match="expected exactly one resolution"):
            bdf.add_bar_close_column(df)

    def test_non_bar_resolution_raises(self):
        df = frame({"date": [t(9, 0)], "resolution": ["1_TICK"]})

        with pytest.raises(ValueError, match="'1_TICK' is not a bar resolution"):
            bdf.add_bar_close_column(df)


class TestValidateSpineDf:
    def test_valid_spine_passes(self, spine: nw.DataFrame[Any]):
        bdf.validate_spine_df(spine)

    def test_missing_key_col_raises(self, spine: nw.DataFrame[Any]):
        with pytest.raises(
            ValueError, match=r"bar spine is missing key columns \['product'\]"
        ):
            bdf.validate_spine_df(spine.drop("product"))

    def test_null_key_raises(self):
        df = frame(
            {"date": [t(9, 0), t(9, 1)], "product": ["BTC", None], "resolution": [R, R]}
        )

        with pytest.raises(
            ValueError, match=r"cannot contain null values: \['product'\]"
        ):
            bdf.validate_spine_df(df)

    def test_duplicate_keys_raise(self):
        df = frame(
            {"date": [t(9, 0)] * 2, "product": ["BTC"] * 2, "resolution": [R] * 2}
        )

        with pytest.raises(ValueError, match="must be unique"):
            bdf.validate_spine_df(df)

    def test_multiple_resolutions_raise(self):
        df = frame(
            {
                "date": [t(9, 0)] * 2,
                "product": ["BTC"] * 2,
                "resolution": [R, "2_MINUTE"],
            }
        )

        with pytest.raises(ValueError, match="exactly one primary resolution"):
            bdf.validate_spine_df(df)


class TestAlignDfToSpine:
    def test_unsupported_mode_raises(self, spine: nw.DataFrame[Any]):
        with pytest.raises(ValueError, match="unsupported alignment mode 'nearest'"):
            bdf.align_df_to_spine(
                spine,
                spine,
                mode="nearest",  # ty: ignore[invalid-argument-type]
            )

    def test_missing_time_col_raises(self, spine: nw.DataFrame[Any]):
        with pytest.raises(
            ValueError, match="contributor dataframe is missing 'news_date'"
        ):
            bdf.align_df_to_spine(spine, spine, mode="event", time_col="news_date")

    @pytest.mark.parametrize("mode", ["exact", "event", "asof"])
    def test_invalid_spine_raises_in_every_mode(
        self, spine: nw.DataFrame[Any], mode: str
    ):
        with pytest.raises(ValueError, match="bar spine is missing key columns"):
            bdf.align_df_to_spine(
                spine.drop("resolution"),
                spine,
                mode=mode,  # ty: ignore[invalid-argument-type]
            )


class TestAlignExact:
    def test_keeps_only_bars_present_on_both_sides(self, spine: nw.DataFrame[Any]):
        contributor = frame(
            {
                "signal": [3, 9, 2, 1],
                "date": [t(9, 3), t(9, 2), t(9, 1), t(9, 0)],
                "product": ["BTC", "BTC", "ETH", "BTC"],
                "resolution": [R] * 4,
            }
        )

        aligned = bdf.align_df_to_spine(spine, contributor, mode="exact")

        assert aligned.columns == [*KEYS, "signal"]
        assert rows(aligned) == [
            (t(9, 0), "BTC", R, 1),
            (t(9, 1), "ETH", R, 2),
            (t(9, 3), "BTC", R, 3),
        ]

    def test_other_resolution_does_not_match(self, spine: nw.DataFrame[Any]):
        contributor = frame(
            {
                "date": [t(9, 0)],
                "product": ["BTC"],
                "resolution": ["2_MINUTE"],
                "signal": [1],
            }
        )

        assert rows(bdf.align_df_to_spine(spine, contributor, mode="exact")) == []

    def test_key_dtypes_are_cast_to_the_spine(self, spine: nw.DataFrame[Any]):
        contributor = frame(
            {"date": [t(9, 0)], "product": ["BTC"], "resolution": [R], "signal": [1]},
            schema_overrides={"date": pl.Datetime("ms")},
        )

        aligned = bdf.align_df_to_spine(spine, contributor, mode="exact")

        assert aligned.to_native().schema["date"] == pl.Datetime("us")
        assert rows(aligned) == [(t(9, 0), "BTC", R, 1)]

    def test_missing_key_col_raises(self, spine: nw.DataFrame[Any]):
        contributor = frame({"date": [t(9, 0)], "product": ["BTC"], "signal": [1]})

        with pytest.raises(ValueError, match=r"missing key columns \['resolution'\]"):
            bdf.align_df_to_spine(spine, contributor, mode="exact")

    def test_duplicate_contributor_keys_raise(self, spine: nw.DataFrame[Any]):
        contributor = frame(
            {
                "date": [t(9, 0)] * 2,
                "product": ["BTC"] * 2,
                "resolution": [R] * 2,
                "signal": [1, 2],
            }
        )

        with pytest.raises(
            ValueError, match=r"exact-aligned contributor keys .* must be unique"
        ):
            bdf.align_df_to_spine(spine, contributor, mode="exact")


class TestAlignEvents:
    def align(
        self,
        spine: nw.DataFrame[Any],
        contributor: nw.DataFrame[Any],
        *,
        time_col: str = "news_date",
        broadcast_null_products: bool = False,
    ) -> nw.DataFrame[Any]:
        return bdf.align_df_to_spine(
            spine,
            contributor,
            mode="event",
            time_col=time_col,
            broadcast_null_products=broadcast_null_products,
        )

    @pytest.mark.parametrize(
        ("event_time", "bar"),
        [
            pytest.param(t(8, 59, 59, 999_999), None, id="before-first-bar"),
            pytest.param(t(9, 0), t(9, 0), id="exactly-on-open"),
            pytest.param(t(9, 0, 59, 999_000), t(9, 0), id="exactly-on-close"),
            pytest.param(t(9, 0, 59, 999_500), t(9, 0), id="inside-last-millisecond"),
            pytest.param(t(9, 1), t(9, 1), id="exactly-on-next-open"),
            pytest.param(
                t(9, 1, 59, 999_999), t(9, 1), id="last-microsecond-before-gap"
            ),
            pytest.param(t(9, 2, 30), None, id="inside-gap"),
            pytest.param(
                t(9, 3, 59, 999_999), t(9, 3), id="last-microsecond-of-last-bar"
            ),
            pytest.param(t(9, 4), None, id="after-last-bar"),
        ],
    )
    def test_event_lands_in_the_bar_whose_half_open_interval_contains_it(
        self, spine: nw.DataFrame[Any], event_time: datetime, bar: datetime | None
    ):
        aligned = self.align(spine, events([event_time], ["BTC"]))

        expected = [] if bar is None else [(bar, "BTC", R, event_time, "h0")]
        assert rows(aligned) == expected

    def test_named_time_col_is_kept_as_the_first_value_col(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = frame(
            {"headline": ["a"], "product": ["BTC"], "news_date": [t(9, 0, 20)]}
        )

        aligned = self.align(spine, contributor)

        assert aligned.columns == [*KEYS, "news_date", "headline"]

    def test_default_date_time_col_is_replaced_by_the_bar_date(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = events([t(9, 0, 20)], ["BTC"], time_col="date")

        aligned = self.align(spine, contributor, time_col="date")

        assert aligned.columns == [*KEYS, "headline"]
        assert rows(aligned) == [(t(9, 0), "BTC", R, "h0")]

    def test_events_within_a_bar_are_ordered_by_time(self, spine: nw.DataFrame[Any]):
        contributor = events([t(9, 1, 50), t(9, 0, 20), t(9, 1, 10)], ["BTC"] * 3)

        aligned = self.align(spine, contributor)

        assert rows(aligned) == [
            (t(9, 0), "BTC", R, t(9, 0, 20), "h1"),
            (t(9, 1), "BTC", R, t(9, 1, 10), "h2"),
            (t(9, 1), "BTC", R, t(9, 1, 50), "h0"),
        ]

    def test_events_within_a_bar_are_ordered_by_time_with_default_time_col(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = events([t(9, 1, 50), t(9, 1, 10)], ["BTC"] * 2, time_col="date")

        aligned = self.align(spine, contributor, time_col="date")

        assert aligned.get_column("headline").to_list() == ["h1", "h0"]

    def test_event_only_matches_bars_of_its_own_product(self, spine: nw.DataFrame[Any]):
        # ETH has no 09:03 bar and SOL is not on the spine at all.
        contributor = events(
            [t(9, 0, 10), t(9, 3, 10), t(9, 0, 10)], ["ETH", "ETH", "SOL"]
        )

        aligned = self.align(spine, contributor)

        assert rows(aligned) == [(t(9, 0), "ETH", R, t(9, 0, 10), "h0")]

    def test_null_product_is_dropped_without_broadcast(self, spine: nw.DataFrame[Any]):
        contributor = events([t(9, 0, 10), t(9, 0, 20)], ["BTC", None])

        aligned = self.align(spine, contributor)

        assert rows(aligned) == [(t(9, 0), "BTC", R, t(9, 0, 10), "h0")]

    def test_null_product_is_broadcast_to_products_with_a_matching_bar(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = events(
            [t(9, 0, 10), t(9, 3, 10), t(9, 2, 10)], [None, None, None]
        )

        aligned = self.align(spine, contributor, broadcast_null_products=True)

        assert rows(aligned) == [
            (t(9, 0), "BTC", R, t(9, 0, 10), "h0"),
            (t(9, 0), "ETH", R, t(9, 0, 10), "h0"),
            (t(9, 3), "BTC", R, t(9, 3, 10), "h1"),
        ]

    def test_product_and_broadcast_events_are_combined(self, spine: nw.DataFrame[Any]):
        contributor = events([t(9, 0, 30), t(9, 0, 10)], ["ETH", None])

        aligned = self.align(spine, contributor, broadcast_null_products=True)

        assert rows(aligned) == [
            (t(9, 0), "BTC", R, t(9, 0, 10), "h1"),
            (t(9, 0), "ETH", R, t(9, 0, 10), "h1"),
            (t(9, 0), "ETH", R, t(9, 0, 30), "h0"),
        ]

    def test_absent_product_col_is_broadcast(self, spine: nw.DataFrame[Any]):
        contributor = events([t(9, 1, 10)])

        aligned = self.align(spine, contributor, broadcast_null_products=True)

        assert rows(aligned) == [
            (t(9, 1), "BTC", R, t(9, 1, 10), "h0"),
            (t(9, 1), "ETH", R, t(9, 1, 10), "h0"),
        ]

    def test_no_match_returns_an_empty_frame_with_the_output_schema(
        self, spine: nw.DataFrame[Any]
    ):
        aligned = self.align(spine, events([t(9, 1, 10)]))

        assert aligned.to_native().schema == {
            "date": pl.Datetime("us"),
            "product": pl.String,
            "resolution": pl.String,
            "news_date": pl.Datetime("us"),
            "headline": pl.String,
        }
        assert len(aligned) == 0

    def test_empty_contributor_returns_an_empty_frame(self, spine: nw.DataFrame[Any]):
        aligned = self.align(spine, events([t(9, 0, 10)], ["BTC"]).head(0))

        assert aligned.columns == [*KEYS, "news_date", "headline"]
        assert len(aligned) == 0

    def test_contributor_date_and_resolution_cols_do_not_override_the_bar_keys(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = frame(
            {
                "news_date": [t(9, 0, 20), t(9, 1, 20)],
                "product": ["BTC", None],
                "date": [datetime(2020, 1, 1)] * 2,
                "resolution": ["1_TICK"] * 2,
                "headline": ["a", "b"],
            },
            schema_overrides={"product": pl.String},
        )

        aligned = self.align(spine, contributor, broadcast_null_products=True)

        assert rows(aligned) == [
            (t(9, 0), "BTC", R, t(9, 0, 20), "a"),
            (t(9, 1), "BTC", R, t(9, 1, 20), "b"),
            (t(9, 1), "ETH", R, t(9, 1, 20), "b"),
        ]

    def test_event_time_is_cast_to_the_spine_date_dtype(self, spine: nw.DataFrame[Any]):
        contributor = frame(
            {"news_date": [t(9, 0, 20)], "product": ["BTC"], "headline": ["a"]},
            schema_overrides={"news_date": pl.Datetime("ms")},
        )

        aligned = self.align(spine, contributor)

        assert aligned.to_native().schema["news_date"] == pl.Datetime("us")
        assert rows(aligned) == [(t(9, 0), "BTC", R, t(9, 0, 20), "a")]

    def test_value_cols_named_like_internal_temporaries_survive(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = frame(
            {
                "news_date": [t(9, 0, 20)],
                "product": ["BTC"],
                "__bar_close": ["keep-1"],
                "__source_time": ["keep-2"],
            }
        )

        aligned = self.align(spine, contributor)

        assert rows(aligned) == [(t(9, 0), "BTC", R, t(9, 0, 20), "keep-1", "keep-2")]


class TestAlignAsof:
    def htf(self, opens: list[datetime], values: list[float]) -> nw.DataFrame[Any]:
        return frame(
            {
                "date": opens,
                "product": ["BTC"] * len(opens),
                "resolution": ["2_MINUTE"] * len(opens),
                "htf_close": values,
            }
        )

    def test_higher_timeframe_bar_is_used_only_after_it_closes(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = self.htf([t(8, 58), t(9, 0), t(9, 2)], [10.0, 20.0, 30.0])

        aligned = bdf.align_df_to_spine(spine, contributor, mode="asof")

        assert aligned.columns == [*KEYS, "htf_close"]
        assert rows(aligned) == [
            # The 09:00 2m bar is still open at the 09:00 1m close.
            (t(9, 0), "BTC", R, 10.0),
            (t(9, 0), "ETH", R, None),
            (t(9, 1), "BTC", R, 20.0),
            (t(9, 1), "ETH", R, None),
            (t(9, 3), "BTC", R, 30.0),
        ]

    def test_null_before_any_contributor_bar_closes_then_carried_forward(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = self.htf([t(9, 0)], [20.0])

        aligned = bdf.align_df_to_spine(spine, contributor, mode="asof")

        assert aligned.filter(nw.col("product") == "BTC").get_column(
            "htf_close"
        ).to_list() == [None, 20.0, 20.0]

    def test_same_resolution_bar_is_available_on_its_own_spine_bar(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = frame(
            {
                "date": [t(9, 1), t(9, 0)],
                "product": ["BTC", "BTC"],
                "resolution": [R, R],
                "signal": [2, 1],
            }
        )

        aligned = bdf.align_df_to_spine(spine, contributor, mode="asof")

        assert aligned.filter(nw.col("product") == "BTC").get_column(
            "signal"
        ).to_list() == [1, 2, 2]

    def test_every_spine_row_is_kept_with_the_spine_resolution(
        self, spine: nw.DataFrame[Any]
    ):
        aligned = bdf.align_df_to_spine(spine, self.htf([t(9, 0)], [20.0]), mode="asof")

        assert rows(aligned.select(KEYS)) == rows(spine.select(KEYS).sort(KEYS))

    def test_key_dtypes_are_cast_to_the_spine(self, spine: nw.DataFrame[Any]):
        contributor = frame(
            {
                "date": [t(8, 58)],
                "product": ["BTC"],
                "resolution": ["2_MINUTE"],
                "htf_close": [10.0],
            },
            schema_overrides={"date": pl.Datetime("ms")},
        )

        aligned = bdf.align_df_to_spine(spine, contributor, mode="asof")

        assert aligned.filter(nw.col("product") == "BTC").get_column(
            "htf_close"
        ).to_list() == [10.0, 10.0, 10.0]

    @pytest.mark.parametrize("missing", ["product", "resolution"])
    def test_missing_required_col_raises(self, spine: nw.DataFrame[Any], missing: str):
        contributor = self.htf([t(9, 0)], [20.0]).drop(missing)

        with pytest.raises(ValueError, match="as-of alignment requires columns"):
            bdf.align_df_to_spine(spine, contributor, mode="asof")

    def test_mixed_contributor_resolutions_raise(self, spine: nw.DataFrame[Any]):
        contributor = frame(
            {
                "date": [t(9, 0), t(9, 0)],
                "product": ["BTC", "ETH"],
                "resolution": ["2_MINUTE", "5_MINUTE"],
                "htf_close": [1.0, 2.0],
            }
        )

        with pytest.raises(ValueError, match="expected exactly one resolution"):
            bdf.align_df_to_spine(spine, contributor, mode="asof")


class TestAggregateEventsByBar:
    def aligned(self) -> nw.DataFrame[Any]:
        return frame(
            {
                "date": [t(9, 1), t(9, 0), t(9, 1), t(9, 0)],
                "product": ["BTC", "ETH", "BTC", "BTC"],
                "resolution": [R] * 4,
                "news_date": [t(9, 1, 50), t(9, 0, 5), t(9, 1, 10), t(9, 0, 20)],
                "headline": ["c", "d", "b", "a"],
            }
        )

    def test_collapses_to_one_row_per_bar_preserving_input_order(self):
        aggregated = bdf.aggregate_events_by_bar(
            self.aligned(), events_col="news_events", count_col="news_count"
        )

        assert aggregated.columns == [*KEYS, "news_events", "news_count"]
        assert rows(aggregated) == [
            (t(9, 0), "BTC", R, [{"news_date": t(9, 0, 20), "headline": "a"}], 1),
            (t(9, 0), "ETH", R, [{"news_date": t(9, 0, 5), "headline": "d"}], 1),
            (
                t(9, 1),
                "BTC",
                R,
                [
                    {"news_date": t(9, 1, 50), "headline": "c"},
                    {"news_date": t(9, 1, 10), "headline": "b"},
                ],
                2,
            ),
        ]

    def test_result_joins_onto_the_spine_without_duplicating_bars(
        self, spine: nw.DataFrame[Any]
    ):
        contributor = events(
            [t(9, 1, 50), t(9, 1, 10), t(9, 0, 20)], ["BTC", "BTC", None]
        )
        aligned = bdf.align_df_to_spine(
            spine,
            contributor,
            mode="event",
            time_col="news_date",
            broadcast_null_products=True,
        )

        aggregated = bdf.aggregate_events_by_bar(
            aligned, events_col="news_events", count_col="news_count"
        )
        joined = spine.join(aggregated, on=KEYS, how="left").sort(KEYS)

        assert len(joined) == len(spine)
        assert joined.get_column("news_count").to_list() == [1, 1, 2, None, None]

    def test_missing_key_col_raises(self):
        with pytest.raises(
            ValueError, match="aligned event dataframe is missing key columns"
        ):
            bdf.aggregate_events_by_bar(
                self.aligned().drop("product"), events_col="e", count_col="n"
            )

    def test_no_event_value_cols_raises(self):
        with pytest.raises(ValueError, match="without event value columns"):
            bdf.aggregate_events_by_bar(
                self.aligned().select(KEYS), events_col="e", count_col="n"
            )

    @pytest.mark.parametrize(
        ("events_col", "count_col", "conflict"),
        [("headline", "n", "headline"), ("e", "news_date", "news_date")],
    )
    def test_output_col_colliding_with_an_event_col_raises(
        self, events_col: str, count_col: str, conflict: str
    ):
        with pytest.raises(ValueError, match=rf"already exist: \['{conflict}'\]"):
            bdf.aggregate_events_by_bar(
                self.aligned(), events_col=events_col, count_col=count_col
            )

    def test_unsupported_backend_raises(self):
        pa = pytest.importorskip("pyarrow")
        table = nw.from_native(self.aligned().to_native().to_arrow(), eager_only=True)
        assert isinstance(table.to_native(), pa.Table)

        with pytest.raises(TypeError, match="supports only Polars and pandas backends"):
            bdf.aggregate_events_by_bar(table, events_col="e", count_col="n")


class TestTemporaryCol:
    def test_returns_the_base_name_when_free(self):
        assert bdf._temporary_col("__tmp", frame({"a": [1]})) == "__tmp"

    def test_prefixes_underscores_until_unused_across_all_frames(self):
        name = bdf._temporary_col(
            "__tmp", frame({"__tmp": [1]}), frame({"___tmp": [1]})
        )

        assert name == "____tmp"


class TestPandasBackend:
    """Each scenario must give pandas the same rows as Polars."""

    @pytest.fixture(autouse=True)
    def require_pandas(self) -> None:
        pytest.importorskip("pandas")

    @staticmethod
    def to_pandas(df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        return nw.from_native(df.to_native().to_pandas(), eager_only=True)

    @staticmethod
    def pandas_rows(df: nw.DataFrame[Any]) -> list[tuple[Any, ...]]:
        return pl.from_pandas(df.to_native()).rows()

    def assert_same(
        self, pandas_result: nw.DataFrame[Any], polars_result: nw.DataFrame[Any]
    ) -> None:
        assert pandas_result.implementation.is_pandas()
        assert pandas_result.columns == polars_result.columns
        assert self.pandas_rows(pandas_result) == rows(polars_result)

    def test_pivot_and_unpivot(self, spine: nw.DataFrame[Any]):
        df = spine.with_columns((nw.col("close") * 2).alias("volume"))
        pivot_cols = ["product", "resolution"]

        wide_pd = bdf.pivot_long_to_wide(
            self.to_pandas(df), index_col="date", pivot_cols=pivot_cols
        )
        wide_pl = bdf.pivot_long_to_wide(df, index_col="date", pivot_cols=pivot_cols)
        long_pd = bdf.unpivot_wide_to_long(
            wide_pd, index_col="date", pivot_cols=pivot_cols
        )

        assert set(wide_pd.columns) == set(wide_pl.columns)
        self.assert_same(wide_pd.select(wide_pl.columns), wide_pl)
        self.assert_same(long_pd, df.sort(KEYS))

    def test_add_bar_close_column(self, spine: nw.DataFrame[Any]):
        self.assert_same(
            bdf.add_bar_close_column(self.to_pandas(spine)),
            bdf.add_bar_close_column(spine),
        )

    def test_exact(self, spine: nw.DataFrame[Any]):
        contributor = frame(
            {
                "signal": [3, 9, 2, 1],
                "date": [t(9, 3), t(9, 2), t(9, 1), t(9, 0)],
                "product": ["BTC", "BTC", "ETH", "BTC"],
                "resolution": [R] * 4,
            }
        )

        self.assert_same(
            bdf.align_df_to_spine(
                self.to_pandas(spine), self.to_pandas(contributor), mode="exact"
            ),
            bdf.align_df_to_spine(spine, contributor, mode="exact"),
        )

    @pytest.mark.parametrize("time_col", ["news_date", "date"])
    @pytest.mark.parametrize("broadcast", [False, True])
    def test_event(self, spine: nw.DataFrame[Any], time_col: str, broadcast: bool):
        contributor = events(
            [
                t(8, 59, 59, 999_999),
                t(9, 1, 50),
                t(9, 0, 59, 999_500),
                t(9, 1, 10),
                t(9, 2, 30),
                t(9, 3, 59, 999_999),
                t(9, 0, 10),
                t(9, 3, 10),
                t(9, 0, 30),
                t(9, 0, 5),
            ],
            ["BTC", "BTC", "BTC", "BTC", "BTC", "BTC", None, None, "ETH", "SOL"],
            time_col=time_col,
        )

        def align(s: nw.DataFrame[Any], c: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
            return bdf.align_df_to_spine(
                s,
                c,
                mode="event",
                time_col=time_col,
                broadcast_null_products=broadcast,
            )

        polars_result = align(spine, contributor)

        assert len(polars_result) == (8 if broadcast else 5)
        self.assert_same(
            align(self.to_pandas(spine), self.to_pandas(contributor)), polars_result
        )

    def test_event_without_product_col(self, spine: nw.DataFrame[Any]):
        contributor = events([t(9, 1, 10), t(9, 2, 10)])

        def align(s: nw.DataFrame[Any], c: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
            return bdf.align_df_to_spine(
                s,
                c,
                mode="event",
                time_col="news_date",
                broadcast_null_products=True,
            )

        self.assert_same(
            align(self.to_pandas(spine), self.to_pandas(contributor)),
            align(spine, contributor),
        )

    def test_event_with_no_match_returns_an_empty_frame(self, spine: nw.DataFrame[Any]):
        aligned = bdf.align_df_to_spine(
            self.to_pandas(spine),
            self.to_pandas(events([t(9, 2, 30)], ["BTC"])),
            mode="event",
            time_col="news_date",
        )

        assert aligned.columns == [*KEYS, "news_date", "headline"]
        assert len(aligned) == 0

    def test_asof(self, spine: nw.DataFrame[Any]):
        contributor = frame(
            {
                "date": [t(9, 2), t(8, 58), t(9, 0)],
                "product": ["BTC"] * 3,
                "resolution": ["2_MINUTE"] * 3,
                "htf_close": [30.0, 10.0, 20.0],
            }
        )

        self.assert_same(
            bdf.align_df_to_spine(
                self.to_pandas(spine), self.to_pandas(contributor), mode="asof"
            ),
            bdf.align_df_to_spine(spine, contributor, mode="asof"),
        )

    def test_aggregate_events_by_bar(self, spine: nw.DataFrame[Any]):
        contributor = events(
            [t(9, 1, 50), t(9, 1, 10), t(9, 0, 20)], ["BTC", "BTC", None]
        )

        def aggregate(s: nw.DataFrame[Any], c: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
            aligned = bdf.align_df_to_spine(
                s,
                c,
                mode="event",
                time_col="news_date",
                broadcast_null_products=True,
            )
            return bdf.aggregate_events_by_bar(
                aligned, events_col="news_events", count_col="news_count"
            )

        pandas_result = aggregate(self.to_pandas(spine), self.to_pandas(contributor))

        self.assert_same(pandas_result, aggregate(spine, contributor))
        joined = self.to_pandas(spine).join(pandas_result, on=KEYS, how="left")
        assert len(joined) == len(spine)
