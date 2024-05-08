# NOTE: need this to make TYPE_CHECKING work to avoid the circular import issue
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.models.model_base import MachineLearningModel
    from pfund.types.core import tModel
    from pfund.models.model_base import BaseModel
    from pfund.datas.data_base import BaseData

import numpy as np
try:
    import pandas as pd
    import polars as pl
except ImportError:
    pass

from pfund.models.model_base import BaseFeature
from pfund.strategies.strategy_base import BaseStrategy
from pfund.mixins.backtest import BacktestMixin


def BacktestModel(Model: type[tModel], ml_model: MachineLearningModel, *args, **kwargs) -> BacktestMixin | tModel:
    class _BacktestModel(BacktestMixin, Model):
        # __getattr__ at this level to get the correct model name
        def __getattr__(self, attr):
            '''gets triggered only when the attribute is not found'''
            try:
                return super().__getattr__(attr)
            except AttributeError:
                class_name = Model.__name__
                raise AttributeError(f"'{class_name}' object or '{class_name}.ml_model' or '{class_name}.data_tool' has no attribute '{attr}'")
        
        def to_dict(self):
            model_dict = super().to_dict()
            model_dict['class'] = Model.__name__
            model_dict['model_signature'] = self._model_signature
            model_dict['data_signatures'] = self._data_signatures
            return model_dict

        def add_consumer(self, consumer: BaseStrategy | BaseModel):
            is_dummy_strategy = isinstance(consumer, BaseStrategy) and consumer.name == '_dummy'
            if is_dummy_strategy:
                assert not self._consumers, f"{self.name} must have _dummy strategy as its only consumer"
            return super().add_consumer(consumer)
        
        def _check_if_dummy_strategy(self):
            if self._consumers:
                # NOTE: dummy strategy will always be the only consumer if it's added
                consumer = self._consumers[0]
                return isinstance(consumer, BaseStrategy) and consumer.name == '_dummy'
            else:
                return False
        
        def on_start(self):
            if self.engine.mode == 'vectorized':
                self.set_group_data(False)
            if self._is_signal_df_required:
                if self._signal_df is None:
                    raise ValueError(
                        f"Please make sure '{self.name}' was dumped "
                        f"using '{self.type}.dump(signal_df)' correctly."
                        # FIXME: correct the link
                        f"Please refer to the doc: https://pfund.ai"  
                    )
                # TODO: check if the signal_df is consistent with the current datas
                else:
                    pass
            super().on_start()
        
        def load(self) -> dict:
            obj: dict = super().load()
            signal_df = obj.get('signal_df', None)
            self.set_signal_df(signal_df)
            return obj

        def dump(self, signal_df: pd.DataFrame | pl.LazyFrame):
            obj = {'signal_df': signal_df}
            super().dump(obj)
            
        def clear_dfs(self):
            assert self.engine.mode == 'event_driven'
            if not self._is_signal_df_required:
                self._data_tool.clear_df()
            for model in self.models.values():
                model.clear_dfs()
            
        # FIXME: pandas specific
        def assert_consistent_signals(self):
            '''Asserts consistent model signals from vectorized and event-driven backtesting, triggered in event-driven backtesting'''
            import pandas.testing as pdt
            event_driven_signal = self.signal_df
            # set signal_df to None and load the vectorized_signal
            self.set_signal_df(None)
            self.load()
            assert self.signal_df is not None, f"Please dump your model '{self.name}' by calling model.dump() before running event-driven backtesting"
            vectorized_signal = self.signal_df
            # filter out the last date since event_driven_signal doesn't have it 
            vectorized_signal_ts_index = vectorized_signal.index.get_level_values('ts')
            last_date = vectorized_signal_ts_index.max()
            vectorized_signal = vectorized_signal[vectorized_signal_ts_index != last_date]

            for col in vectorized_signal.columns:
                pdt.assert_series_equal(vectorized_signal[col], event_driven_signal[col], check_exact=False, rtol=1e-5)
            
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