from abc import ABCMeta


class MetaStrategy(ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        module_name = namespace.get('__module__', '')
        is_user_defined_class = (
            not module_name.startswith('pfund.') and
            not module_name.startswith('ray.')
        )
        if is_user_defined_class:
            original_init = cls.__init__  # capture before overwrite
            def init_in_correct_order(self, *args, **kwargs):
                # force to init the BaseClass first
                BaseClass = cls.__bases__[0]
                BaseClass.__init__(self, *args, **kwargs)
                cls.__original_init__(self, *args, **kwargs)
            cls.__init__ = init_in_correct_order
            # during Ray's pickling, somehow its called twice, need to check if __original_init__ is already set
            if not hasattr(cls, '__original_init__'):
                cls.__original_init__ = original_init
        return cls
    
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        
        # FIXME: update backtest strategy
        if name == '_BacktestStrategy':
            assert '__init__' not in dct, '_BacktestStrategy should not have __init__()'
    
    # def __call__(cls, *args, **kwargs):
    #     instance = super().__call__(*args, **kwargs)
    #     return instance