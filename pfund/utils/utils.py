from typing import Any

import os
import importlib
import inspect
import datetime
from pathlib import Path


class Singleton:
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]
    
    @classmethod
    def _remove_singleton(cls):
        if cls in cls._instances:
            del cls._instances[cls]


# used to explicitly mark a function that includes an api call
def is_api_call(func):
    return func


def is_command_available(cmd):
    import shutil
    return shutil.which(cmd) is not None


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def convert_to_uppercases(*args):
    return (s.upper() if type(s) is str else s for s in args)


def convert_to_lowercases(*args):
    return (s.lower() if type(s) is str else s for s in args)


def convert_ts_to_dt(ts: float):
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)

def get_local_timezone() -> datetime.timezone:
    return datetime.datetime.now().astimezone().tzinfo

def lowercase(func):
    """Convert all args and kwargs to lowercases, used as a decorator"""
    def wrapper(*args, **kwargs):
        args = (arg.lower() if type(arg) is str else arg for arg in args)
        kwargs = {k: v.lower() if type(v) is str else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper


def load_yaml_file(file_path) -> dict | list[dict]:
    import yaml
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            contents = list(yaml.safe_load_all(f))
            if not contents:
                return {}
            elif len(contents) == 1:
                return contents[0]
            else:
                return contents
    else:
        return {}


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_telegram_bot_updates(token):
    import requests
    url = f'https://api.telegram.org/bot{token}/getUpdates'
    ret = requests.get(url)
    try:
        return ret.json()
    except:
        return ret


def get_last_modified_time(file_path: str) -> datetime.datetime:
    # Get the last modified time in seconds since epoch
    last_modified_time = os.path.getmtime(file_path)
    # Convert to datetime object
    return datetime.datetime.fromtimestamp(last_modified_time, tz=datetime.timezone.utc)


def parse_api_response_with_schema(response: dict, schema: dict) -> list[dict]:
    """
    Parse API response according to schema definition.
    
    The schema supports:
    1. Direct string values (hardcoded values)
    2. List/tuple paths with optional transformers
    3. Nested dictionary schemas
    
    Returns:
        List[dict]: Always returns a list of parsed dictionaries for consistency,
                   even if input contains only a single item
    """
    # Get the data to parse based on 'result' path
    # result_path is e.g. ['result', 'list'], meaning that the data to parse is under 'result' and 'list'
    result_path = schema.get('result', [])
    data = response
    
    for key in result_path:
        data = data[key]
    
    # Convert single dict to list[dict] for consistency
    if isinstance(data, dict):
        data = [data]
    
    # Remove 'result' from schema since it's just a path indicator
    parse_schema = {k: v for k, v in schema.items() if k != 'result'}
    
    def parse_single_item(item: dict) -> dict:
        output = {}
        for key, value_path in parse_schema.items():
            if isinstance(value_path, str):
                # Case 1: Hardcoded value
                output[key] = value_path
            elif isinstance(value_path, (list, tuple)):
                # Case 2: Path with optional transformers
                current_value = item
                for transformer in value_path:
                    if isinstance(transformer, str):
                        if transformer in current_value:
                            current_value = current_value[transformer]
                        else:
                            break
                    else:
                        # Apply function transformer
                        current_value = transformer(current_value)
                output[key] = current_value
            elif isinstance(value_path, dict):
                # Case 3: Nested schema
                nested_output = {}
                for nested_key, nested_value_path in value_path.items():
                    current_value = item
                    for transformer in nested_value_path:
                        if isinstance(transformer, str):
                            current_value = current_value[transformer]
                        else:
                            current_value = transformer(current_value)
                    nested_output[nested_key] = current_value
                output[key] = nested_output
        return output
    
    # Parse all items and always return a list
    return [parse_single_item(item) for item in data]


def find_strategy_class(strat: str):
    from pfund.strategies.strategy_base import BaseStrategy
    module = importlib.import_module(f'pfund.strategies.{strat.lower()}.strategy')
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            return obj
    return None


def short_path(path: str, last_n_parts: int=3) -> Path:
    parts = Path(path).parts[-last_n_parts:]
    return Path(*parts)


def get_duplicate_functions(obj: object, obj2: object):
    '''Returns duplicate functions between obj and obj2'''
    obj_funcs = {func_name for func_name in dir(obj) if callable(getattr(obj, func_name)) and not (func_name.startswith('__') and func_name.endswith('__'))}
    obj2_funcs = {func_name for func_name in dir(obj2) if callable(getattr(obj2, func_name)) and not (func_name.startswith('__') and func_name.endswith('__'))}
    return obj_funcs & obj2_funcs


def get_function_signature(function: object, without_self=True) -> inspect.Signature:
    '''
    without_self: 
        if False, signature will be e.g. (self, a, b, c)
        if True, signature will be e.g. (a, b, c)
    '''
    signature = inspect.signature(function)
    if without_self:
        params_without_self = [param for name, param in signature.parameters.items() if name != 'self']
        signature = inspect.signature(function).replace(parameters=params_without_self)
    return signature


def get_args_and_kwargs_from_function(function: object) -> tuple[list[str], list[tuple[str, Any]], str | None, str | None]:
    signature = get_function_signature(function)
    args = []
    kwargs: list[tuple[str, Any]] = []  # [(name, default_value)]
    var_args = var_kwargs = None
    # Iterate over the parameters of the signature
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # This is *args
            var_args = name
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # This is **kwargs
            var_kwargs = name
        elif param.default is inspect.Parameter.empty:
            # Regular positional argument
            args.append(name)
        else:
            # Keyword argument with a default value
            kwargs.append((name, param.default))
    return args, kwargs, var_args, var_kwargs
