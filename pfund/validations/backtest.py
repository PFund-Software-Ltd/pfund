import functools

from pydantic import TypeAdapter
from pfeed.const.common import SUPPORTED_DATA_FEEDS
from pfeed import ALIASES as PFEED_ALIASES

from pfund.types.backtest import BacktestKwargs
from pfund.utils.utils import get_engine_class


def validate_backtest_kwargs(func):
    Engine = get_engine_class()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'backtest' not in kwargs or not kwargs['backtest']:
            raise Exception(f'kwargs "backtest" is missing or empty in {func.__name__}{args[1:]}')
        else:
            backtest_kwargs = kwargs['backtest']
            if 'data_source' in backtest_kwargs:
                backtest_kwargs['data_source'] = backtest_kwargs['data_source'].upper()
                if backtest_kwargs['data_source'] in PFEED_ALIASES:
                    backtest_kwargs['data_source'] = PFEED_ALIASES[backtest_kwargs['data_source']]
            else:
                trading_venue = args[0].upper()
                backtest_kwargs['data_source'] = trading_venue
            assert backtest_kwargs['data_source'] in SUPPORTED_DATA_FEEDS, f"{backtest_kwargs['data_source']=} not in {SUPPORTED_DATA_FEEDS}"
            backtest_kwargs_adapter = TypeAdapter(BacktestKwargs)
            backtest_kwargs_adapter.validate_python(backtest_kwargs)
            has_date_range = 'start_date' in backtest_kwargs and 'end_date' in backtest_kwargs
            has_rollback = 'rollback_period' in backtest_kwargs
            if has_date_range == has_rollback:
                raise ValueError("Please provide either ('start_date', 'end_date') or 'rollback_period' in kwargs 'backtest', but not both")
        if Engine.env == 'TRAIN':
            if 'train' not in kwargs or not kwargs['train']:
                raise Exception(f'kwargs "train" is missing or empty in {func.__name__}{args[1:]}')
        return func(*args, **kwargs)
    
    return wrapper