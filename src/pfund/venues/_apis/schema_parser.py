from __future__ import annotations
from typing import Any, TypeAlias
from collections.abc import Sequence, Callable

from pydantic import validate_call

from pfund.venues._apis.typing import RawResponse, Schema, ResponseData

from pfund.errors import ResponseParseError


# --- extraction-rule vocabulary (parser-internal) ---
# An "extraction rule" is a schema value. It is one of three shapes:
#   1. FieldPath  — a list/tuple mixing Extractors (dict keys to descend into)
#                   and Transformers (callables applied to the extracted value),
#                   e.g. ('timestamp', lambda ms: ms / 1000) or ['price', float]
#   2. Schema     — a nested schema (dict); its source is located via an "@key"
#   3. a hardcoded value (anything else) — emitted verbatim
Extractor: TypeAlias = str
Transformer: TypeAlias = Callable[[Any], Any]
FieldPath: TypeAlias = Sequence["Extractor | Transformer"]
# the value of an "@key": the path locating a nested schema's source. Like a
# FieldPath, it mixes Extractors (dict keys to descend) and Transformers
# (callables applied to the current value), e.g. ('result', 'list', lambda x: x[0]).
SourcePath: TypeAlias = Sequence["Extractor | Transformer"]


# OPTIMIZE
class SchemaParser:
    """
    Transforms API responses into standardized format using schema-based rules.

    **Core Concept:**
    A schema defines how to convert API responses into standardized dictionaries.
    Normal fields extract directly from the root message, while nested schemas
    use special `@xxx` keys to locate their data source.

    **Schema Structure:**
    ```python
    schema = {
        'field1': extraction_rule,        # Extract from root message
        'field2': extraction_rule,        # Extract from root message
        '@nested_key': ['path', 'to'],    # Path for nested schema (only when needed)
        'nested_key': {                   # Nested schema definition
            'sub_field': extraction_rule
        }
    }
    ```

    **Extraction Rules:**

    1. **Hardcoded values** (non list/tuple/dict types):
       ```python
       'category': 'spot'  # Always outputs this exact value
       ```

    2. **Field paths** (list/tuple with extractors and transformers):
       ```python
       'price': ['current_price', float]           # Extract nested field, convert to float
       'ts': ('timestamp', lambda ms: ms / 1000)   # Extract and transform timestamp
       ```

    3. **Nested schemas** (dict with corresponding @xxx path):
       ```python
       '@data': ['response', 'items'],  # Path to nested data
       'data': {                        # Schema for nested data
           'open': ('open', float),
           'close': ('close', float)
       }
       ```

    **Processing Logic:**

    - **Normal fields**: Extract directly from the root API response
    - **Nested schemas**: Use `@xxx` keys to locate data, then apply nested schema
    - **Field paths**: Processed left-to-right as pipeline (extractors → transformers)
    - **Missing fields**: Set to None when extractor keys don't exist
    - **Response types**: Handles both single dict and list of dicts automatically

    **Key Features:**

    - No special keys needed for simple schemas
    - `@xxx` keys only required for nested data extraction
    - Graceful handling of optional fields (e.g., 'expiration' for futures only)
    - Automatic handling of single vs. multiple response items

    **Example:**
    ```python
    payload = {
        'topic': 'kline.1.BTCUSDT',
        'ts': 1752760761525,
        'data': [
            {'open': '118407.3', 'close': '118354.3', 'timestamp': 1752760761525}
        ]
    }

    schema = {
        'ts': ('ts', lambda ms: ms / 1000),  # Extract from root, transform
        '@data': ['data'],                   # Path to nested data
        'data': {                           # Nested schema
            'open': ('open', float),
            'close': ('close', float),
            'ts': ('timestamp', float)
        }
    }

    # Output: {
    #     'ts': 1752760761.525,
    #     'data': [{'open': 118407.3, 'close': 118354.3, 'ts': 1752760761525.0}]
    # }
    ```
    """

    @staticmethod
    def _descend(obj: Any, key: str) -> Any:
        """Descend one key into a dict, yielding None for an absent key or a
        null/non-dict parent. Shared by field-path and nested-schema source-path
        traversal so both tolerate missing/null envelope keys identically —
        exchanges routinely drop or null these on empty or degraded responses.
        """
        return obj[key] if isinstance(obj, dict) and key in obj else None

    @staticmethod
    def _walk(value: Any, path: "FieldPath | SourcePath") -> Any:
        """Walk a path left-to-right as a pipeline: string Extractors descend one
        dict key (via _descend), callable Transformers are applied to the current
        value. A missing/null intermediate short-circuits to None before any
        downstream transformer runs. Shared by field-path (Case 1) and nested-schema
        source-path (Case 2) traversal so both honor Extractors *and* Transformers
        identically.
        """
        for step in path:
            if isinstance(step, str):
                extractor: Extractor = step
                value = SchemaParser._descend(value, extractor)
                # missing/null intermediate -> None; stop before any
                # downstream transformer runs on it
                if value is None:
                    return None
            elif isinstance(step, dict):
                raise ValueError(
                    f"dict is not allowed for extractor or transformer: {step}"
                )
            else:
                # NOTE: it doesn't have to be a function, e.g. it could be an operation like float()
                transformer: Transformer = step
                value = transformer(value)
        return value

    @staticmethod
    @validate_call
    def _parse(
        msg: RawResponse | dict[str, Any], schema: Schema | dict[str, Any]
    ) -> dict[str, Any]:
        """
        Convert the input message to the desired format according to the schema.
        """
        try:
            output: dict[str, Any] = {}
            for key, rule in schema.items():
                if key.startswith("@"):
                    continue
                # Case 1: field path — extractors (str keys) and/or transformers (callables)
                if isinstance(rule, (list, tuple)):
                    field_path: FieldPath = rule
                    output[key] = SchemaParser._walk(msg, field_path)
                # Case 2: nested schema (dict), located via its "@key" source path
                elif isinstance(rule, dict):
                    nested_schema = rule
                    source_key = f"@{key}"
                    assert source_key in schema, (
                        f'"{source_key}" must be defined for nested schema "{key}"'
                    )
                    source_path: SourcePath = schema[source_key]
                    source: Any = SchemaParser._walk(msg, source_path)
                    output[key] = (
                        None
                        if source is None
                        else SchemaParser.convert(source, nested_schema)
                    )
                # Case 3: hardcoded value
                else:
                    output[key] = rule
            return output
        except Exception as exc:
            raise ResponseParseError(f"Failed to parse api response: {exc}")

    @staticmethod
    @validate_call
    def convert(
        payload: RawResponse | dict[str, Any] | list[dict[str, Any]],
        schema: Schema | dict[str, Any],
    ) -> ResponseData:
        if isinstance(payload, dict):
            return SchemaParser._parse(payload, schema)
        elif isinstance(payload, list):
            return [SchemaParser._parse(item, schema) for item in payload]
        else:
            raise ResponseParseError(
                f'Unhandled API response type "{type(payload)}":\n{payload}'
            )
