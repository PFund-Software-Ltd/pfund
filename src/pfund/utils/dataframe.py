from typing import Any

import narwhals as nw


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
