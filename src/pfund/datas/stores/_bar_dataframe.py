"""Bar-centric dataframe transformations and alignment.

Bar dataframes use the canonical long-form key ``(date, product, resolution)``.
A spine contains the receiving bar universe; contributors are aligned to it
without being allowed to add rows. Precise events map to their containing bar,
while bar contributors align by the time their bars become available.

These operations are stateless. Callers must pass the spine explicitly.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

import narwhals as nw

from pfund.datas.resolution import Resolution


BarAlignmentMode = Literal["exact", "event", "asof"]


INDEX_COL = "date"
PIVOT_COLS = ["product", "resolution"]
KEY_COLS = [INDEX_COL, *PIVOT_COLS]


def reorder_key_cols(df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
    """Move the canonical bar key columns to the left."""
    key_cols = KEY_COLS
    return df.select(key_cols + [col for col in df.columns if col not in key_cols])


def _encode_pivot_component(col: str) -> nw.Expr:
    return (
        nw.col(col)
        .cast(nw.String)
        .str.replace_all("%", "%25", literal=True)
        .str.replace_all("\x00", "%00", literal=True)
        .str.replace_all(":", "%3A", literal=True)
    )


def _decode_pivot_component(value: str) -> str:
    # Decode escape sequences before "%25" so an original literal such as
    # "%3A" (encoded as "%253A") is not mistaken for an encoded colon.
    return value.replace("%3A", ":").replace("%00", "\x00").replace("%25", "%")


def pivot_long_to_wide(
    df: nw.DataFrame[Any],
    *,
    index_col: str,
    pivot_cols: list[str],
) -> nw.DataFrame[Any]:
    """Pivots a long-form dataframe to wide form with flat, reference-safe names.

    Each pivoted value column is named ``{pivot_key}:{field}``, where ``pivot_key``
    joins the ``pivot_cols`` values with ":" in ``pivot_cols`` order, e.g.
    ``1_MINUTE:BYBIT_BTC_USDT_PERPETUAL:close``. This replaces polars' default
    ``close_{"1_MINUTE","BYBIT_BTC_USDT_PERPETUAL"}`` naming, which embeds
    quotes/commas that break direct column references.

    The combination of ``index_col`` and ``pivot_cols`` must be a non-null,
    unique composite key.

    Args:
        df: dataframe in long form
        index_col: the column to keep as the wide-form index (e.g. "date")
        pivot_cols: non-null string columns folded into the wide-form column names
    """
    key_cols = [index_col, *pivot_cols]
    null_key_cols = [col for col in key_cols if df.get_column(col).null_count() > 0]
    if null_key_cols:
        raise ValueError(
            f"composite key columns cannot contain null values: {null_key_cols}"
        )

    schema = df.collect_schema()
    non_string_pivot_cols = [col for col in pivot_cols if schema[col] != nw.String]
    if non_string_pivot_cols:
        raise TypeError(
            "pivot columns must contain string identifiers; "
            + f"non-string columns: {non_string_pivot_cols}"
        )

    if len(df.select(key_cols).unique()) != len(df):
        raise ValueError(f"columns {key_cols} must form a unique composite key")
    value_cols = [col for col in df.columns if col not in key_cols]
    # collapse the multi-column key into one ":"-joined key, so polars emits
    # flat "{field}{sep}{pivot_key}" names instead of its {"a","b"} tuple form
    pivot_key_col = "__pivot_key__"
    while pivot_key_col in df.columns:
        pivot_key_col = f"_{pivot_key_col}"
    df = df.with_columns(
        nw.concat_str(
            [_encode_pivot_component(col) for col in pivot_cols],
            separator=":",
        ).alias(pivot_key_col)
    )
    pivot_keys = df.get_column(pivot_key_col).unique().to_list()
    # NUL separator: never collides with field names that contain "_"
    # (e.g. n_data_points), so the rename below is unambiguous
    sep = "\x00"
    wide = df.pivot(
        on=pivot_key_col,
        index=index_col,
        values=value_cols,
        separator=sep,
        sort_columns=True,
    )
    if len(value_cols) == 1:
        field = value_cols[0]
        rename = {pivot_key: f"{pivot_key}:{field}" for pivot_key in pivot_keys}
    else:
        rename = {
            f"{field}{sep}{pivot_key}": f"{pivot_key}:{field}"
            for field in value_cols
            for pivot_key in pivot_keys
        }
    return wide.rename(rename).sort(index_col)


def unpivot_wide_to_long(
    df: nw.DataFrame[Any],
    *,
    index_col: str,
    pivot_cols: list[str],
) -> nw.DataFrame[Any]:
    """Reverse a dataframe produced by :func:`pivot_long_to_wide`.

    Encoded pivot identifiers are decoded back into ``pivot_cols`` and value
    fields regain their original column names. All-null wide combinations are
    omitted because they represent combinations absent from the original long
    dataframe. Consequently, exact round-tripping requires every original long
    row to contain at least one non-null value field.

    Args:
        df: dataframe in the wide form produced by ``pivot_long_to_wide``
        index_col: the wide-form index column (e.g. "date")
        pivot_cols: names to restore for the encoded pivot-key components
    """
    if not pivot_cols:
        raise ValueError("pivot_cols must contain at least one column")
    if index_col not in df.columns:
        raise ValueError(f"index column {index_col!r} not found in dataframe")

    wide_cols = [col for col in df.columns if col != index_col]
    if not wide_cols:
        raise ValueError("wide dataframe has no pivoted value columns")

    columns_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    value_cols: list[str] = []
    for wide_col in wide_cols:
        parts = wide_col.split(":", len(pivot_cols))
        if len(parts) != len(pivot_cols) + 1:
            raise ValueError(f"column {wide_col!r} is not a valid pivoted column")
        pivot_values = tuple(_decode_pivot_component(value) for value in parts[:-1])
        value_col = parts[-1]
        if value_col not in value_cols:
            value_cols.append(value_col)
        fields = columns_by_key.setdefault(pivot_values, {})
        if value_col in fields:
            raise ValueError(
                f"duplicate value field {value_col!r} for pivot key {pivot_values}"
            )
        fields[value_col] = wide_col

    expected_value_cols = set(value_cols)
    for pivot_values, fields in columns_by_key.items():
        missing_value_cols = expected_value_cols - set(fields)
        if missing_value_cols:
            raise ValueError(
                f"pivot key {pivot_values} is missing value fields: "
                + f"{sorted(missing_value_cols)}"
            )

    long_dfs: list[nw.DataFrame[Any]] = []
    for pivot_values, fields in columns_by_key.items():
        long_df = df.select(
            [index_col] + [nw.col(fields[field]).alias(field) for field in value_cols]
        ).with_columns(
            nw.lit(value).alias(col)
            for col, value in zip(pivot_cols, pivot_values, strict=True)
        )
        long_df = long_df.select([index_col, *pivot_cols, *value_cols])

        has_value = ~nw.col(value_cols[0]).is_null()
        for value_col in value_cols[1:]:
            has_value = has_value | ~nw.col(value_col).is_null()
        long_dfs.append(long_df.filter(has_value))

    return nw.concat(long_dfs, how="vertical").sort([index_col, *pivot_cols])


def add_bar_close_column(
    df: nw.DataFrame[Any],
    *,
    date_col: str = "date",
    resolution_col: str = "resolution",
    output_col: str = "_bar_close",
) -> nw.DataFrame[Any]:
    """Add close-time availability for a single-resolution bar dataframe."""
    missing_cols = [col for col in (date_col, resolution_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"cannot calculate bar close time; missing columns {missing_cols}"
        )

    resolution_values = df.get_column(resolution_col)
    if resolution_values.null_count() > 0:
        raise ValueError(
            "cannot calculate bar close time; "
            + f"{resolution_col!r} cannot contain null values"
        )
    resolutions = resolution_values.unique().to_list()
    if len(resolutions) != 1:
        raise ValueError(
            "cannot calculate bar close time; expected exactly one resolution, "
            + f"found {resolutions}"
        )

    value = resolutions[0]
    resolution = Resolution(value)
    if not resolution.is_bar():
        raise ValueError(f"{value!r} is not a bar resolution")
    offset = timedelta(
        seconds=resolution.to_seconds(),
        milliseconds=-1,
    )
    return df.with_columns((nw.col(date_col) + offset).alias(output_col))


def align_df_to_spine(
    spine_df: nw.DataFrame[Any],
    contributor_df: nw.DataFrame[Any],
    *,
    mode: BarAlignmentMode,
    time_col: str = "date",
    broadcast_null_products: bool = False,
) -> nw.DataFrame[Any]:
    """Align contributor rows to the canonical long-form bar spine.

    Args:
        spine_df: The canonical receiving bar spine.
        contributor_df: Secondary dataframe to align.
        mode: Temporal alignment policy:
            - ``exact`` requires the same bar keys as the spine.
            - ``event`` maps precise timestamps to their containing bars.
            - ``asof`` uses the latest contributor bar available by each
              spine bar's close time.
        time_col: Contributor timestamp column. Event callers should rename
            a generic ``date`` to a semantic name such as ``news_date`` if
            the precise source timestamp must remain in the output.
        broadcast_null_products: In event mode, expand rows with a null or
            absent product across products present in the matching bar.

    Returns:
        Contributor values keyed by the canonical bar keys. The spine's own
        value columns are not included.
    """
    validate_spine_df(spine_df)
    _validate_contributor_df(contributor_df, time_col=time_col)

    if mode == "exact":
        return _align_exact(spine_df, contributor_df)
    if mode == "event":
        return _align_events(
            spine_df,
            contributor_df,
            time_col=time_col,
            broadcast_null_products=broadcast_null_products,
        )
    if mode == "asof":
        return _align_asof(spine_df, contributor_df, time_col=time_col)
    raise ValueError(f"unsupported alignment mode {mode!r}")


def aggregate_events_by_bar(
    aligned_df: nw.DataFrame[Any],
    *,
    events_col: str,
    count_col: str,
) -> nw.DataFrame[Any]:
    """Collapse aligned event rows to one row per bar.

    ``aligned_df`` may contain multiple events with the same
    ``(date, product, resolution)`` key. Every non-key column is treated as
    part of an event record, and the records are collected into one list so
    the result can be joined onto a unique bar spine without duplicating
    bar rows. Event order from ``aligned_df`` is preserved within each list.

    Args:
        aligned_df: Event rows already aligned to the canonical bar keys.
        events_col: Name of the output list column, such as
            ``"news_events"``. Its elements are structs with a Polars
            backend and dictionaries with a pandas backend.
        count_col: Name of the output event-count column, such as
            ``"news_count"``.

    Returns:
        A dataframe containing the bar keys, ``events_col``, and
        ``count_col``, with one row for each bar that has at least one
        aligned event. Bars without events are added later when this result
        is left-joined onto the bar spine.
    """
    _validate_key_cols(
        aligned_df,
        KEY_COLS,
        dataframe_name="aligned event dataframe",
    )
    event_cols = [col for col in aligned_df.columns if col not in KEY_COLS]
    if not event_cols:
        raise ValueError("cannot aggregate events without event value columns")
    conflicting_cols = [
        col for col in (events_col, count_col) if col in aligned_df.columns
    ]
    if conflicting_cols:
        raise ValueError(f"event output columns already exist: {conflicting_cols}")

    native_df = aligned_df.to_native()
    native_module = type(native_df).__module__.split(".", maxsplit=1)[0]
    if native_module == "polars":
        import polars as pl

        aggregated = (
            native_df.group_by(KEY_COLS, maintain_order=True)
            .agg(
                pl.struct(event_cols).alias(events_col),
                pl.len().alias(count_col),
            )
            .sort(KEY_COLS)
        )
    elif native_module == "pandas":
        import pandas as pd

        records_col = _temporary_col(
            "__event_record",
            aligned_df,
        )
        events_df = native_df.copy()
        events_df[records_col] = events_df[event_cols].to_dict(orient="records")
        aggregated = (
            events_df.groupby(
                KEY_COLS,
                as_index=False,
                sort=True,
                dropna=False,
            )[records_col]
            .agg(list)
            .rename(columns={records_col: events_col})
        )
        aggregated[count_col] = aggregated[events_col].map(len)
        aggregated = pd.DataFrame(aggregated)
    else:
        raise TypeError(
            "event aggregation supports only Polars and pandas backends; "
            + f"got {type(native_df)!r}"
        )
    return nw.from_native(aggregated)


def validate_spine_df(spine_df: nw.DataFrame[Any]) -> None:
    """Validate the canonical key and resolution invariants of a bar spine."""
    _validate_key_cols(
        spine_df,
        KEY_COLS,
        dataframe_name="bar spine",
        require_non_null=True,
        require_unique=True,
    )

    resolutions = spine_df.get_column("resolution").unique().to_list()
    if len(resolutions) != 1:
        raise ValueError(
            "bar spine must contain exactly one primary resolution; "
            + f"found {resolutions}"
        )


def _validate_contributor_df(
    contributor_df: nw.DataFrame[Any],
    *,
    time_col: str,
) -> None:
    if time_col not in contributor_df.columns:
        raise ValueError(f"contributor dataframe is missing {time_col!r}")


def _validate_key_cols(
    df: nw.DataFrame[Any],
    key_cols: list[str],
    *,
    dataframe_name: str,
    require_non_null: bool = False,
    require_unique: bool = False,
) -> None:
    """Validate a dataframe's required key-column invariants."""
    missing_key_cols = [col for col in key_cols if col not in df.columns]
    if missing_key_cols:
        raise ValueError(f"{dataframe_name} is missing key columns {missing_key_cols}")
    if require_non_null:
        null_key_cols = [col for col in key_cols if df.get_column(col).null_count() > 0]
        if null_key_cols:
            raise ValueError(
                f"{dataframe_name} key columns cannot contain null values: "
                + f"{null_key_cols}"
            )
    if require_unique and len(df.select(key_cols).unique()) != len(df):
        raise ValueError(f"{dataframe_name} keys {key_cols} must be unique")


def _align_exact(
    spine_df: nw.DataFrame[Any],
    contributor_df: nw.DataFrame[Any],
) -> nw.DataFrame[Any]:
    _validate_key_cols(
        contributor_df,
        KEY_COLS,
        dataframe_name="exact-aligned contributor",
        require_unique=True,
    )

    spine_schema = spine_df.collect_schema()
    contributor_df = contributor_df.with_columns(
        nw.col(col).cast(spine_schema[col]).alias(col) for col in KEY_COLS
    )
    value_cols = [col for col in contributor_df.columns if col not in KEY_COLS]
    return (
        spine_df.select(KEY_COLS)
        .join(contributor_df, on=KEY_COLS, how="inner")
        .select(KEY_COLS + value_cols)
        .sort(KEY_COLS)
    )


def _align_events(
    spine_df: nw.DataFrame[Any],
    contributor_df: nw.DataFrame[Any],
    *,
    time_col: str,
    broadcast_null_products: bool,
) -> nw.DataFrame[Any]:
    close_col = _temporary_col("__bar_close", spine_df, contributor_df)
    source_time_col = _temporary_col("__source_time", spine_df, contributor_df)
    spine = add_bar_close_column(
        spine_df.select(KEY_COLS),
        output_col=close_col,
    )
    spine_schema = spine_df.collect_schema()
    contributor = contributor_df.rename({time_col: source_time_col}).with_columns(
        nw.col(source_time_col).cast(spine_schema[INDEX_COL]).alias(source_time_col)
    )
    has_product = "product" in contributor.columns
    if has_product:
        contributor = contributor.with_columns(
            nw.col("product").cast(spine_schema["product"]).alias("product")
        )
        product_events = contributor.filter(~nw.col("product").is_null())
    else:
        product_events = contributor.head(0)

    value_cols = [
        col
        for col in contributor_df.columns
        if col not in KEY_COLS and col != "product"
    ]
    if time_col not in KEY_COLS:
        value_cols = [time_col, *[col for col in value_cols if col != time_col]]

    aligned_dfs: list[nw.DataFrame[Any]] = []
    if len(product_events):
        aligned_dfs.append(
            _join_events_to_intervals(
                product_events,
                spine,
                source_time_col=source_time_col,
                close_col=close_col,
                by="product",
            )
        )

    if broadcast_null_products:
        generic_events = (
            contributor.filter(nw.col("product").is_null()).drop("product")
            if has_product
            else contributor
        )
        if len(generic_events):
            intervals = spine.select([INDEX_COL, "resolution", close_col]).unique()
            aligned_generic = _join_events_to_intervals(
                generic_events,
                intervals,
                source_time_col=source_time_col,
                close_col=close_col,
                by=None,
            )
            aligned_generic = aligned_generic.join(
                spine.select(KEY_COLS),
                on=[INDEX_COL, "resolution"],
                how="inner",
            )
            aligned_dfs.append(aligned_generic)

    if not aligned_dfs:
        contributor_schema = contributor_df.collect_schema()
        return (
            spine_df.select(KEY_COLS)
            .head(0)
            .with_columns(
                nw.lit(None).cast(contributor_schema[col]).alias(col)
                for col in value_cols
            )
        )

    aligned = nw.concat(aligned_dfs, how="diagonal")
    aligned = aligned.rename({source_time_col: time_col})
    return aligned.select(KEY_COLS + value_cols).sort(
        KEY_COLS + ([time_col] if time_col not in KEY_COLS else [])
    )


def _join_events_to_intervals(
    events: nw.DataFrame[Any],
    intervals: nw.DataFrame[Any],
    *,
    source_time_col: str,
    close_col: str,
    by: str | None,
) -> nw.DataFrame[Any]:
    # pandas requires the as-of key to be globally sorted even when the
    # join is partitioned with ``by``. Time-first sorting also leaves each
    # product partition ordered, which satisfies Polars.
    sort_cols = [source_time_col] + ([by] if by else [])
    interval_sort_cols = [close_col] + ([by] if by else [])
    aligned = events.sort(sort_cols).join_asof(
        intervals.sort(interval_sort_cols),
        left_on=source_time_col,
        right_on=close_col,
        by=by,
        strategy="forward",
    )
    # A forward as-of join could select the next bar across a market gap.
    # Keep only events that actually fall inside the selected bar interval.
    return aligned.filter(
        ~nw.col(INDEX_COL).is_null()
        & (nw.col(source_time_col) >= nw.col(INDEX_COL))
        & (nw.col(source_time_col) <= nw.col(close_col))
    )


def _align_asof(
    spine_df: nw.DataFrame[Any],
    contributor_df: nw.DataFrame[Any],
    *,
    time_col: str,
) -> nw.DataFrame[Any]:
    required_cols = ["product", "resolution"]
    missing_cols = [col for col in required_cols if col not in contributor_df.columns]
    if missing_cols:
        raise ValueError(f"as-of alignment requires columns {missing_cols}")

    close_col = _temporary_col("__bar_close", spine_df, contributor_df)
    available_col = _temporary_col("__available_at", spine_df, contributor_df)
    spine = add_bar_close_column(
        spine_df.select(KEY_COLS),
        output_col=close_col,
    )
    spine_schema = spine_df.collect_schema()
    contributor_df = contributor_df.with_columns(
        nw.col(time_col).cast(spine_schema[INDEX_COL]).alias(time_col),
        nw.col("product").cast(spine_schema["product"]).alias("product"),
    )
    contributor = add_bar_close_column(
        contributor_df,
        date_col=time_col,
        output_col=available_col,
    )
    value_cols = [col for col in contributor.columns if col not in KEY_COLS]
    value_cols = [col for col in value_cols if col != available_col]

    aligned = spine.sort([close_col, "product"]).join_asof(
        contributor.select(["product", available_col, *value_cols]).sort(
            [available_col, "product"]
        ),
        left_on=close_col,
        right_on=available_col,
        by="product",
        strategy="backward",
    )
    return aligned.select(KEY_COLS + value_cols).sort(KEY_COLS)


def _temporary_col(
    base_name: str,
    *dfs: nw.DataFrame[Any],
) -> str:
    existing_cols = {col for df in dfs for col in df.columns}
    col = base_name
    while col in existing_cols:
        col = f"_{col}"
    return col
