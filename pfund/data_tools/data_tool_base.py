from pfund.datas.data_base import BaseData


class BaseDataTool:
    INDEX = ['ts', 'product', 'resolution']
    GROUP = ['product', 'resolution']
    _MAX_NEW_ROWS = 1000
    _MIN_ROWS = 1_000
    _MAX_ROWS = None

    def __init__(self):
        self.train_periods = {}  # {product: ('start_date', 'end_date')}
        self.val_periods = self.validation_periods = {}  # {product: ('start_date', 'end_date')}
        self.test_periods = {}  # {product: ('start_date', 'end_date')}
        self.train_set = None
        self.val_set = self.validation_set = None
        self.test_set = None
        self.df = None
        # used in event-driven looping to avoid appending data to df one by one
        # instead, append data to _new_rows and whenever df is needed,
        # push the data in _new_rows to df
        self._new_rows = []  # [{col: value, ...}]
        self._raw_dfs = {}  # {data: df}
    
    @classmethod
    def set_min_rows(cls, min_rows: int):
        cls._MIN_ROWS = min_rows
    
    @classmethod
    def set_max_rows(cls, max_rows: int):
        cls._MAX_ROWS = max_rows

    def get_raw_df(self, data: BaseData):
        return self._raw_dfs[data]
    
    def has_raw_df(self, data: BaseData):
        return data in self._raw_dfs
    
    def add_raw_df(self, data: BaseData, df):
        self._raw_dfs[data] = df
    
    def set_data_periods(self, datas, **kwargs):
        train_period = kwargs.get('train_period', None)
        val_period = kwargs.get('validation_period', None) or kwargs.get('val_period', None)
        test_period = kwargs.get('test_period', None)
        for data in datas:
            product = data.product
            self.train_periods[product] = train_period
            self.val_periods[product] = self.validation_periods[product] = val_period
            self.test_periods[product] = test_period
