import os
import importlib
import inspect
import datetime
from pathlib import Path

from typing import Any

import requests


class Singleton:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
            # print(f'Singleton class "{cls.__name__}" is created')
        return cls._instance


# @override = the function is only created to override 
# to prevent from calling the parent class's function
def override(fn):
    return fn


# used to explicitly mark a function that includes an api call
def is_api_call(func):
    return func


def convert_to_uppercases(*args):
    return (s.upper() if type(s) is str else s for s in args)


def convert_to_lowercases(*args):
    return (s.lower() if type(s) is str else s for s in args)


def convert_ts_to_dt(ts: float):
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)


def lowercase(func):
    """Convert all args and kwargs to lowercases, used as a decorator"""
    def wrapper(*args, **kwargs):
        args = (arg.lower() if type(arg) is str else arg for arg in args)
        kwargs = {k: v.lower() if type(v) is str else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper


def convert_path_to_Path_obj(path: str | Path) -> Path:
    if type(path) is str:
        return Path(path)
    elif type(path) is Path:
        pass
    else:
        raise ValueError(f'path must be str or Path, not {type(path)}')
    return path


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
    url = f'https://api.telegram.org/bot{token}/getUpdates'
    ret = requests.get(url)
    try:
        return ret.json()
    except:
        return ret


# step over result by following schema
def step_into(ret, schema):
    if isinstance(schema, list):
        for s in schema:
            next_ret = ret[s]
            next_schema = schema[1:]
            if not next_schema:
                return next_ret
            else:  
                return step_into(next_ret, next_schema)
    else:
        if schema is not None:
            return ret.get(schema, schema)
        else:
            return None
        

def find_strategy_class(strat: str):
    from pfund.strategies.strategy_base import BaseStrategy
    module = importlib.import_module(f'pfund.strategies.{strat.lower()}.strategy')
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            return obj
    return None


def get_engine_class():
    from pfund.const.commons import SUPPORTED_ENVIRONMENTS
    env = os.getenv('env')
    assert env in SUPPORTED_ENVIRONMENTS, f'Unsupported {env=}'
    if env == 'BACKTEST':
        from pfund import BacktestEngine as Engine
    elif env == 'TRAIN':
        from pfund import TrainEngine as Engine
    elif env == 'TEST':
        from pfund import TestEngine as Engine
    else:
        from pfund import TradeEngine as Engine
    return Engine


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



def get_args_and_kwargs_from_function(function: object) -> tuple[str]:
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
