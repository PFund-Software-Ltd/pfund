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
                BaseClass.__init__(self)
                original_init(self, *args, **kwargs)
            cls.__init__ = init_in_correct_order
        return cls
    
    # NOTE: both __call__ and __init__ will NOT be called when using Ray,
    def __call__(cls, *args, **kwargs):
        is_backtest = cls.__name__ == '_BacktestStrategy'
        if is_backtest:
            # cls.__bases__ are (BacktestMixin, Strategy)
            _cls = cls.__bases__[1]
            backtest_mixin_cls = cls.__bases__[0]
        else:
            _cls = cls

        instance = super().__call__(*args, **kwargs)
        instance.__pfund_args__ = args
        instance.__pfund_kwargs__ = kwargs

        if is_backtest:
            backtest_mixin_cls.__post_init__(instance, *args, **kwargs)
        
        return instance
    
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        
        if name == '_BacktestStrategy':
            assert '__init__' not in dct, '_BacktestStrategy should not have __init__()'