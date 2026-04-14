from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy import ndarray
    from pfeed.typing import GenericFrame

from abc import abstractmethod

from pfund.components.models.model_base import BaseModel


class BaseFeature(BaseModel):
    '''Feature is a model with model=None'''
    def __init__(self, *args, **kwargs):
        model = None
        super().__init__(model, *args, **kwargs)
        self.set_signal_cols([self.name])
    
    # TODO: generalize return type to dict[str, ndarray] for multi-output features.
    # Today's single-ndarray contract forces a 1-feature-per-class shape, which
    # doesn't fit transforms that naturally emit multiple named outputs together:
    #   - HF tokenizers: {'input_ids', 'attention_mask', 'token_type_ids'}
    #   - StandardScaler on N cols: one scaled col per input col
    #   - OHLCV stat bundles: {'return', 'log_return', 'zscore'}
    # Keep single-ndarray as valid shorthand (auto-wrap into {self.name: arr}) so
    # simple cases stay simple; dict-return is opt-in for multi-output features.
    # Downstream models then need a way to select specific named columns as input.
    @abstractmethod
    def transform(self, X: GenericFrame, *args, **kwargs) -> ndarray:
        """Extract features from the input data"""
        pass
    
    # Create predict alias that calls transform - this maintains compatibility with BaseModel
    def predict(self, X: GenericFrame, *args, **kwargs) -> ndarray:
        return self.transform(X, *args, **kwargs)
    
    def _assert_functions_signatures(self):
        from pfund_kit.utils.function import get_function_args_and_kwargs
        super()._assert_functions_signatures()
        def _assert_predict_function():
            args, kwargs, _, _ = get_function_args_and_kwargs(self.transform)
            if not args or args[0] != 'X':
                raise Exception(f'{self.name} transform() must have "X" as its first arg, i.e. transform(self, X, *args, **kwargs)')
        _assert_predict_function()