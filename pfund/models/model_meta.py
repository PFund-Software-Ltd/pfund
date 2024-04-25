import inspect
from abc import ABCMeta


class MetaModel(ABCMeta):
    def __call__(cls, *args, **kwargs):
        is_backtest = cls.__name__ == '_BacktestModel'
        if is_backtest:
            # cls.__bases__ are (BacktestMixin, Model)
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
        
        if name == '_BacktestModel':
            assert '__init__' not in dct, '_BacktestModel should not have __init__()'
    
        if '__init__' not in dct:
            return
        
        # force users to include 'ml_model'/'indicator' as the first argument in __init__()
        required_arg = ''
        base_names = {b.__name__: b for b in bases}
                
        if 'BaseModel' in base_names:
            if name == 'BaseIndicator':
                pass
            elif name == 'BaseFeature':
                pass
            else:
                required_arg = 'ml_model'
        elif 'BaseIndicator' in base_names or 'TALibIndicator' in base_names or 'TAIndicator' in base_names:
            required_arg = 'indicator'
        
        if required_arg:
            args = cls.__init__.__code__.co_varnames
            if required_arg not in args:
                raise TypeError(f"{name}.__init__() must include an '{required_arg}' argument")
            elif args[1] != required_arg:
                raise TypeError(f"'{required_arg}' argument must be the first argument (next to 'self') in {name}.__init__()")