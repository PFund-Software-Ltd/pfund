import numpy as np
import pandas as pd
from abc import abstractmethod

from pfund.models.model_base import BaseModel


class SKLearnModel(BaseModel):
    @abstractmethod
    def prepare_target(self, *args, **kwargs) -> pd.DataFrame:
        pass
    
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        if type(X) is pd.DataFrame:
            X = X.to_numpy()
        if type(y) is pd.DataFrame:
            y = y.to_numpy()
        return self.ml_model.fit(X, y)
    
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if type(X) is pd.DataFrame:
            X = X.to_numpy()
        pred_y = self.ml_model.predict(X)
        return pred_y
    
    def flow(self, is_dump=True, path: str='') -> pd.DataFrame:
        X: pd.DataFrame = self.prepare_features()
        y: pd.DataFrame = self.prepare_target()
        self.logger.debug(f'training model {self.name}')
        self.fit(X, y)
        self.logger.debug(f'trained model {self.name}')
        pred_y: np.ndarray = self.predict(X)
        signal: pd.DataFrame = self.to_signal(X, pred_y)
        if is_dump:
            self.dump(signal, path=path)
        return signal
