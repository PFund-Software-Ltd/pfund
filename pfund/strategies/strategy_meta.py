import inspect
from abc import ABCMeta


class MetaStrategy(ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        module_name = namespace.get('__module__', '')
        is_user_strategy_class = (
            name != 'BaseStrategy' 
            and not module_name.startswith('pfund.') 
            and not module_name.startswith('ray.')
        )
        if is_user_strategy_class:
            for base in bases:
                if base.__name__ == 'BaseStrategy':
                    # NOTE: since ray's ActorHandle doesn't save the original class, we need to save it manually
                    base.__pfund_strategy_class__ = cls
                    break
        return cls
    
    def __call__(cls, *args, **kwargs):
        is_backtest = cls.__name__ == '_BacktestStrategy'
        if is_backtest:
            # cls.__bases__ are (BacktestMixin, Strategy)
            _cls = cls.__bases__[1]
            backtest_mixin_cls = cls.__bases__[0]
        else:
            _cls = cls
        instance = super().__call__(*args, **kwargs)
        module = inspect.getmodule(_cls)
        is_user_defined_class = not module.__name__.startswith('pfund.')
        has_its_own_init = _cls.__init__ is not super(_cls, _cls).__init__
        has_super_init_inside_its_own_init = '_args' in instance.__dict__ and '_kwargs' in instance.__dict__
        if is_user_defined_class and has_its_own_init and not has_super_init_inside_its_own_init:
            if _cls.__bases__:
                BaseClass = _cls.__bases__[0]
            BaseClass.__init__(instance, *args, **kwargs)
        
        if is_backtest:
            backtest_mixin_cls.__post_init__(instance, *args, **kwargs)
        
        return instance
    
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        
        if name == '_BacktestStrategy':
            assert '__init__' not in dct, '_BacktestStrategy should not have __init__()'