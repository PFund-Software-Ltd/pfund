from typing import Any, Callable

import os
import importlib
import inspect
import datetime
from pathlib import Path

import yaml
from pfund.enums import RunMode


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


def convert_ts_to_dt(ts: float):
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)


def get_local_timezone() -> datetime.timezone:
    return datetime.datetime.now().astimezone().tzinfo


def get_telegram_bot_updates(token):
    import requests
    url = f'https://api.telegram.org/bot{token}/getUpdates'
    ret = requests.get(url)
    try:
        return ret.json()
    except Exception:
        return ret


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
