from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl
    import torch
    from pfund.models.model_base import MachineLearningModel
    from pfund.typing import ModelT

from pfund.models.model_base import BaseFeature
from pfund.mixins.backtest_mixin import BacktestMixin


def BacktestModel(Model: type[ModelT], ml_model: MachineLearningModel, *args, **kwargs) -> BacktestMixin | ModelT:
    class _BacktestModel(BacktestMixin, Model):
        def __getattr__(self, name):
            if hasattr(super(), name):
                return getattr(super(), name)
            else:
                class_name = Model.__name__
                raise AttributeError(f"'{class_name}' object has no attribute '{name}'")

        def on_start(self):
            if self._is_signal_df_required:
                self.set_group_data(False)
            super().on_start()
            
        def flow(self) -> pd.DataFrame | pl.LazyFrame:
            self.logger.warning(
                f"creating '{self.name}' signal_df on the fly: "
                "featurize() -> predict(X) -> signalize(X, pred_y)"
            )
            self.set_group_data(False)
            X: pd.DataFrame | pl.LazyFrame = self.featurize()
            pred_y: torch.Tensor | np.ndarray = self.predict(X)
            signal_df: pd.DataFrame | pl.LazyFrame = self.signalize(X, pred_y)
            return signal_df
        
        def load(self) -> dict:
            obj: dict = super().load()
            if self._is_signal_df_required:
                if signal_df := obj.get('signal_df', None):
                    self.logger.info(f"{self.name}'s signal_df is loaded")
                else:
                    signal_df = self.flow()
                self._set_signal_df(signal_df)
            if self.is_model() and not self._is_dummy_strategy:
                error_msg = (
                    f"please make sure '{self.name}' was dumped "
                    f"using '{self.component_type}.dump()' correctly.\n"
                    # FIXME: correct the link
                    "Please refer to the doc for more details: https://pfund.ai"  
                )
                assert self.ml_model, f"{self.ml_model=}, {error_msg}"
            return obj

        def dump(self, signal_df: pd.DataFrame | pl.LazyFrame):
            super().dump({'signal_df': signal_df})
        
    try:       
        if not issubclass(Model, BaseFeature):
            return _BacktestModel(ml_model, *args, **kwargs)
        else:
            return _BacktestModel(*args, **kwargs)
    except TypeError as e:
        raise TypeError(
            f'if super().__init__() is called in {Model.__name__ }.__init__() (which is unnecssary), '
            'make sure it is called with args and kwargs, i.e. super().__init__(*args, **kwargs)'
        ) from e