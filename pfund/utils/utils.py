from typing import Any, Callable

import os
import importlib
import inspect
import datetime
from pathlib import Path
from enum import StrEnum
from decimal import Decimal

import yaml
from pfund.enums import RunMode

# HACK: need to add this to yaml so that StrEnum can be dumped to yaml
yaml.SafeDumper.add_multi_representer(
    StrEnum,
    yaml.representer.SafeRepresenter.represent_str,
)
yaml.SafeDumper.add_multi_representer(
    Decimal,
    lambda dumper, data: dumper.represent_str(str(data))
)
yaml.SafeDumper.add_multi_representer(
    Path,
    lambda dumper, data: dumper.represent_str(str(data))
)


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


def get_free_port() -> int:
    """Get a free port by binding to port 0 and letting the OS assign one."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]
    

def derive_run_mode(ray_kwargs: dict | None=None) -> RunMode:
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
    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)


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
