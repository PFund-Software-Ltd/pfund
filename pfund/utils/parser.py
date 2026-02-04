from pydantic import validate_call

from pfund.errors import ParseApiResponseError


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
    api_response = {
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
    @validate_call
    def _parse(msg: dict, schema: dict) -> dict:
        '''
        Convert the input message to the desired format according to the schema.
        '''
        try:
            output = {}
            for key, path in schema.items():
                if key.startswith('@'):
                    continue
                # Case 1: Path with optional transformers
                if isinstance(path, (list, tuple)):
                    value = msg
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
                        elif isinstance(extractor_or_transformer, dict):
                            raise ValueError(f'dict is not allowed for extractor or transformer: {extractor_or_transformer}')
                        else:
                            # NOTE: it doesn't have to be a function, e.g. it could be an operation like float()
                            transformer = extractor_or_transformer
                            value = transformer(value)
                    output[key] = value
                # Case 2: Nested schema
                elif isinstance(path, dict):
                    key_to_source = f'@{key}'
                    assert key_to_source in schema, f'"{key_to_source}" must be defined for nested schema'
                    path_to_source = schema[key_to_source]
                    value: dict = msg
                    for p in path_to_source:
                        value: dict | list[dict] = value[p]
                    nested_schema = path
                    output[key] = SchemaParser.convert(value, nested_schema)
                # Case 3: Hardcoded value
                else:
                    output[key] = path
            return output
        except Exception as exc:
            raise ParseApiResponseError(f'Failed to parse api response: {exc}')
    
    @staticmethod
    @validate_call
    def convert(api_response: dict | list[dict], schema: dict) -> dict | list[dict]:
        if isinstance(api_response, dict):
            result: dict =  SchemaParser._parse(api_response, schema)
            return result
        elif isinstance(api_response, list):
            result: list[dict] = [SchemaParser._parse(item, schema) for item in api_response]
            return result
        else:
            raise ParseApiResponseError(f'Unhandled API response type "{type(api_response)}":\n{api_response}')