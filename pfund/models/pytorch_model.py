import os

import torch
import pandas as pd
from abc import abstractmethod

from pfund.models.model_base import BaseModel
from pfund.utils.utils import short_path


class PyTorchModel(BaseModel):
    def __call__(self, *args, **kwargs):
        return self.ml_model(*args, **kwargs)
    
    @abstractmethod
    def prepare_target(self, *args, **kwargs) -> pd.DataFrame:
        pass
    
    def load(self, path: str=''):
        file_path = self._get_file_path(path=path, extension='.pt')
        if os.path.exists(file_path):
            obj = torch.load(file_path)
            self.signal = obj['signal']
            self.ml_model.load_state_dict(obj['state_dict'])
            self._assert_no_missing_datas(obj)
            self.logger.debug(f"loaded trained model '{self.name}' from {short_path(file_path)}")
        else:
            self.logger.debug(f"no trained model '{self.name}' found in {short_path(file_path)}")
    
    def dump(self, signal: torch.Tensor, path: str=''):
        signal = self._convert_string_columns_to_objects(signal)
        obj = {
            'signal': signal,
            "state_dict": self.ml_model.state_dict(),
            'datas': self.datas,
            # TODO, refer to model_base, e.g. dump self.datas
            # 'dataset_periods': {
            #     'train_period': "2020-01-01 to 2020-12-31",
            #     'val_period': ...,
            #     'test_period': ...,
            # },
            # "data_info": {
            #     'data_source': ...,
            #     'resolution': ...,
            # }
        }
        file_path = self._get_file_path(path=path, extension='.pt')
        # TODO: need to dump datasets (parquet.gz) as well?
        torch.save(obj, file_path)
        self.logger.debug(f"dumped trained model '{self.name}' to {short_path(file_path)}")
        
    def predict(self, X: torch.Tensor | pd.DataFrame) -> torch.Tensor:
        if type(X) is pd.DataFrame:
            X = torch.tensor( X.to_numpy(), dtype=torch.float32 )
        pred_y = self.ml_model(X)
        return pred_y

    def flow(self, is_dump=True, path: str='') -> pd.DataFrame:
        X: pd.DataFrame = self.prepare_features()
        y: pd.DataFrame = self.prepare_target()
        if 'fit' not in self.__dict__:
            raise Exception(f"fit() is not found in model '{self.name}'")
        self.logger.debug(f'training model {self.name}')
        self.fit(X, y)
        self.logger.debug(f'trained model {self.name}')
        pred_y: torch.Tensor = self.predict(X)
        signal: pd.DataFrame = self.to_signal(X, pred_y)
        if is_dump:
            self.dump(signal, path=path)
        return signal
