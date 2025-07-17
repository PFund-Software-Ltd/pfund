from pydantic import validate_call

from pfund.errors import ParseApiResponseError


class SchemaParser:
    """
    Transforms API responses into standardized format using schema-based rules.
    
    **Core Concept:**
    A schema defines how to convert exchange-specific API responses into standardized 
    dictionaries. The parser extracts data from the API response and applies 
    transformation rules to produce consistent output format.
    
    **Schema Structure:**
    ```python
    schema = {
        '@result': ['path', 'to', 'data'],  # Locates data within API response
        'output_field': extraction_rule,    # Maps to standardized field names
        # ... more field mappings
    }
    ```
    
    **Extraction Rules:**
    
    1. **Hardcoded values** (non list/tuple/dict type, e.g. string):
       ```python
       'category': 'spot'  # Always outputs this exact value
       ```
    
    2. **Field paths** (list/tuple with extractors and transformers):
       ```python
       'price': ['priceData', 'current', float]  # Extract nested field, convert to float
       'ts': ('timestamp', lambda ms: ms / 1000)  # Extract and transform timestamp
       ```
    
    3. **Nested schemas** (dict):
       ```python
       'data': {
           'open': ('open', float),
           'close': ('close', float)
       }
       ```
    
    **Field Path Processing:**
    Field paths are processed left-to-right as a pipeline:
    - **Extractors** (strings): Navigate into nested data structures using dict keys
    - **Transformers** (functions): Modify the extracted value
    
    **Missing Field Handling:**
    If an extractor key doesn't exist in the data, the field is set to None and 
    processing continues. This allows schemas to include fields that only apply 
    to certain data types (e.g., 'expiration' for futures but not perpetuals).
    
    **Input/Output:**
    - **Input**: API response (dict or list of dicts) + schema definition
    - **Output**: List of standardized dictionaries (always returns list for consistency)
    - The '@result' key is removed from the schema after locating the data
    
    **Example:**
    ```python
    api_response = {
        'data': [
            {'symbol': 'BTCUSD', 'price': '50000', 'volume': '100'},
            {'symbol': 'ETHUSD', 'price': '3000', 'volume': '200'}
        ]
    }
    
    schema = {
        '@result': ['data'],
        'product': ['symbol'],
        'price': ('price', float),
        'volume': ('volume', float)
    }
    
    # Output: [
    #     {'product': 'BTCUSD', 'price': 50000.0, 'volume': 100.0},
    #     {'product': 'ETHUSD', 'price': 3000.0, 'volume': 200.0}
    # ]
    ```
    """
    result_key: str = "@result"  # used to locate the result to be parsed
    
    @classmethod
    @validate_call
    def _parse(cls, item: dict, schema: dict) -> dict:
        '''Parse the item according to the schema
        Args:
            item: The item to be parsed
            schema: The schema to be used for parsing
        Returns:
            The parsed item
        '''
        try:
            parsed_item = {}
            for key, path in schema.items():
                if key == cls.result_key:
                    continue
                # Case 1: Path with optional transformers
                if isinstance(path, (list, tuple)):
                    value = item
                    for extractor_or_transformer in path:
                        if isinstance(extractor_or_transformer, str):
                            extractor: str = extractor_or_transformer
                            assert isinstance(value, dict), f'Value is not a dict: {value}'
                            # Graceful handling: Set None for missing keys (schema fields may not apply to all data types)
                            if extractor in value:
                                value = value[extractor]
                            else:
                                value = None
                                break
                        else:
                            # NOTE: it doesn't have to be a function, e.g. it could be an operation like float()
                            transformer = extractor_or_transformer
                            value = transformer(value)
                    parsed_item[key] = value
                # Case 2: Nested schema
                # REVIEW: nested schema cannot use @result key, so now assume that "item" is already the result
                elif isinstance(path, dict):
                    nested_schema = path
                    parsed_item[key] = cls._parse(item, nested_schema)
                # Case 3: Hardcoded value
                else:
                    parsed_item[key] = path
            return parsed_item
        except Exception as exc:
            raise ParseApiResponseError(f'Failed to parse api response: {exc}')
    
    @classmethod
    @validate_call
    def _extract_result(cls, api_response: dict, result_path: list[str] | tuple[str]) -> dict | list[dict]:
        '''Extract the result to be parsed from the API response'''
        try:
            result_to_be_parsed: dict = api_response
            for key in result_path:
                result_to_be_parsed = result_to_be_parsed[key]
            return result_to_be_parsed
        except Exception as exc:
            raise ParseApiResponseError(f'Failed to extract result: {exc}')
    
    @classmethod
    @validate_call
    def convert(cls, api_response: dict, schema: dict) -> dict | list[dict]:
        '''
        Convert the API response to the desired format according to the schema.
        Args:
            api_response: API response
            schema: Schema definition
        '''
        result_to_be_parsed: dict | list[dict] = cls._extract_result(api_response, schema[cls.result_key])
        if isinstance(result_to_be_parsed, dict):
            return cls._parse(result_to_be_parsed, schema)
        elif isinstance(result_to_be_parsed, list):
            return [cls._parse(item, schema) for item in result_to_be_parsed]
        else:
            raise ParseApiResponseError(f'Unhandled result type "{type(result_to_be_parsed)}":\n{result_to_be_parsed}')