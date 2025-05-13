from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy import ndarray
    from pfeed.typing import GenericFrame

from abc import abstractmethod

from pfund.models.model_base import BaseModel


class BaseFeature(BaseModel):
    '''Feature is a model with model=None'''
    def __init__(self, *args, **kwargs):
        model = None
        super().__init__(model, *args, **kwargs)
        self._set_signal_cols([self.name])
    
    @abstractmethod
    def extract(self, X: GenericFrame, *args, **kwargs) -> ndarray:
        """Extract features from the input data"""
        pass
    
    # Create predict alias that calls extract - this maintains compatibility with BaseModel
    def predict(self, X: GenericFrame, *args, **kwargs) -> ndarray:
        return self.extract(X, *args, **kwargs)
    
    def _assert_functions_signatures(self):
        from pfund.utils.utils import get_args_and_kwargs_from_function
        super()._assert_functions_signatures()
        def _assert_predict_function():
            args, kwargs, _, _ = get_args_and_kwargs_from_function(self.extract)
            if not args or args[0] != 'X':
                raise Exception(f'{self.name} extract() must have "X" as its first arg, i.e. extract(self, X, *args, **kwargs)')
        _assert_predict_function()