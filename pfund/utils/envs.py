import os
import functools


def env_wrapper(func, envs: list[str]):
    envs = [env.upper() for env in envs]
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv('env') in envs:
            return func(*args, **kwargs)
        else:
            raise Exception(f"{func.__name__}() is only available in environments '{envs}'.")
    return wrapper


def backtest(func):
    return env_wrapper(func, envs=['BACKTEST'])


def trade(func):
    return env_wrapper(func, envs=['TEST', 'PAPER', 'LIVE'])
    

def live(func):
    return env_wrapper(func, envs=['LIVE'])