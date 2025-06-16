from typing import Any, Callable

import os
import importlib
import inspect
import datetime
from pathlib import Path


from pfund.enums import RunMode


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


def derive_run_mode(ray_kwargs: dict) -> RunMode:
    from mtflow.utils.utils import is_wasm
    if is_wasm():
        run_mode = RunMode.WASM
        assert not ray_kwargs, 'Ray is not supported in WASM mode'
    else:
        if ray_kwargs:
            # NOTE: if `num_cpus` is not set, Ray will only use 1 CPU for scheduling, and 0 CPU for running
            assert 'num_cpus' in ray_kwargs, '`num_cpus` must be set for a Ray actor'
            assert ray_kwargs['num_cpus'] > 0, '`num_cpus` must be greater than 0'
            run_mode = RunMode.REMOTE
        else:
            run_mode = RunMode.LOCAL
    return run_mode


# used to explicitly mark a function that includes an api call
def is_api_call(func):
    return func


def is_command_available(cmd):
    import shutil
    return shutil.which(cmd) is not None


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


def load_yaml_file(file_path) -> dict | list[dict] | None:
    import yaml
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        contents = list(yaml.safe_load_all(f))
        if not contents:
            return {}
        elif len(contents) == 1:
            return contents[0]
        else:
            return contents


def dump_yaml_file(file_path: str, data: dict | list[dict]):
    import yaml
    with open(file_path, 'w') as f:
        yaml.dump(data, f)


def get_telegram_bot_updates(token):
    import requests
    url = f'https://api.telegram.org/bot{token}/getUpdates'
    ret = requests.get(url)
    try:
        return ret.json()
    except Exception:
        return ret


def get_last_modified_time(file_path: str) -> datetime.datetime:
    # Get the last modified time in seconds since epoch
    last_modified_time = os.path.getmtime(file_path)
    # Convert to datetime object
    return datetime.datetime.fromtimestamp(last_modified_time, tz=datetime.timezone.utc)


def parse_raw_result(result: dict | list[dict], schema: dict) -> list[dict]:
    """
    Parse API returned raw result according to schema definition.
    
    The schema supports:
    1. Direct string values (hardcoded values)
    2. List/tuple paths with optional transformers
    3. Nested dictionary schemas
    
    Returns:
        List[dict]: Always returns a list of parsed dictionaries for consistency,
                   even if input contains only a single item
    """
    # Get the result to parse based on 'result' path
    # result_path is e.g. ['result', 'list'], meaning that the result to parse is under 'result' and 'list'
    result_path = schema.get('result', [])
    
    for key in result_path:
        result = result[key]
    
    # Convert single dict to list[dict] for consistency
    if isinstance(result, dict):
        result = [result]
    
    # Remove 'result' from schema since it's just a path indicator
    parse_schema = {k: v for k, v in schema.items() if k != 'result'}

    def parse_single_item(item: dict) -> dict:
        output = {}
        for key, value_path in parse_schema.items():
            # Case 1: Path with optional transformers
            if isinstance(value_path, (list, tuple)):
                current_value = item
                for transformer in value_path:
                    if isinstance(transformer, str):
                        if transformer in current_value:
                            current_value = current_value[transformer]
                        else:
                            current_value = None
                            break
                    else:
                        # Apply function transformer
                        current_value = transformer(current_value)
                output[key] = current_value
            # Case 2: Nested schema
            elif isinstance(value_path, dict):
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
            # Case 3: Hardcoded value
            else:
                output[key] = value_path
        return output
    
    # Parse all items and always return a list
    return [parse_single_item(item) for item in result]


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


def get_function_signature(function: object, skip_self_and_cls: bool=True) -> inspect.Signature:
    """
    Returns the function signature.
    
    If skip_self_and_cls is True, removes 'self' and 'cls' from the parameters.
    """
    signature = inspect.signature(function)
    if skip_self_and_cls:
        filtered_params = [
            param for name, param in signature.parameters.items() if name not in ('self', 'cls')
        ]
        signature = signature.replace(parameters=filtered_params)
    return signature


def get_args_and_kwargs_from_function(
    function: Callable,
    skip_self_and_cls: bool=True,
) -> tuple[list[str], dict[str, Any], str | None, str | None]:
    """
    Parses the function's signature into:
    - a list of required positional/keyword arguments (without defaults),
    - a dict of keyword arguments with default values,
    - the name of *args if present,
    - the name of **kwargs if present.
    """
    signature = get_function_signature(function, skip_self_and_cls=skip_self_and_cls)
    args: list[str] = []
    kwargs: dict[str, Any] = {}
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
            kwargs[name] = param.default
    return args, kwargs, var_args, var_kwargs
