from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
if TYPE_CHECKING:
    from sklearn.model_selection._split import BaseCrossValidator
    from pfeed.enums import DataTool
    from pfeed._typing import GenericDataFrame
    from pfund._typing import DataRangeDict, DatasetSplitsDict
    from pfund.datas.data_base import BaseData


class BaseDataTool:
    name: ClassVar[DataTool]
    
    INDEX = ['date', 'product', 'resolution']
    GROUP = ['product', 'resolution']
    _MAX_NEW_ROWS = 1000
    _MIN_ROWS = 1_000
    _MAX_ROWS = None
    

    def __init__(self):
        # Ensure the child class has defined `name`
        if not hasattr(type(self), 'name'):
            raise AttributeError(f"{self.__class__.__name__} must define a class variable `name`")
        self.df = None
        # used in event-driven looping to avoid appending data to df one by one
        # instead, append data to _new_rows and whenever df is needed,
        # push the data in _new_rows to df
        self._new_rows = []  # [{col: value, ...}]
    
     # FIXME: to be removed
    # def prepare_datasets(self, datas):
    #     # create datasets based on train/val/test periods
    #     datasets = defaultdict(list)  # {'train': [df_of_product_1, df_of_product_2]}
    #     for product in datas:
    #         for type_, periods in [('train', self.train_periods), ('val', self.val_periods), ('test', self.test_periods)]:
    #             period = periods[product]
    #             if period is None:
    #                 raise Exception(f'{type_}_period for {product} is not specified')
    #             df = self.filter_df(self.df, start_date=period[0], end_date=period[1], symbol=product.symbol).reset_index()
    #             datasets[type_].append(df)
                
    #     # combine datasets from different products to create the final train/val/test set
    #     for type_ in ['train', 'val', 'test']:
    #         df = pd.concat(datasets[type_])
    #         df.set_index(self.INDEX, inplace=True)
    #         df.sort_index(level='date', inplace=True)
    #         if type_ == 'train':
    #             self.train_set = df
    #         elif type_ == 'val':
    #             self.val_set = self.validation_set = df
    #         elif type_ == 'test':
    #             self.test_set = df

    def __str__(self):
        return self.name.value

    @classmethod
    def set_min_rows(cls, min_rows: int):
        cls._MIN_ROWS = min_rows
    
    @classmethod
    def set_max_rows(cls, max_rows: int):
        cls._MAX_ROWS = max_rows
