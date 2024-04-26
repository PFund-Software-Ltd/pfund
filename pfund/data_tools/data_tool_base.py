from pfund.datas.data_base import BaseData


class BaseDataTool:
    def __init__(self):
        self.train_periods = {}  # {product: ('start_date', 'end_date')}
        self.val_periods = self.validation_periods = {}  # {product: ('start_date', 'end_date')}
        self.test_periods = {}  # {product: ('start_date', 'end_date')}
        self.train_set = None
        self.val_set = self.validation_set = None
        self.test_set = None
        self.df = None
        self._raw_dfs = {}  # {data: df}
    
    def get_raw_df(self, data: BaseData):
        return self._raw_dfs[data]
    
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
