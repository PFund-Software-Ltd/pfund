# NOTE: need this to make TYPE_CHECKING work to avoid the circular import issue
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.models.model_base import MachineLearningModel
    from pfund.types.core import tModel
    from pfund.models.model_base import BaseModel
    from pfund.datas.data_base import BaseData

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
                assert not self._consumers, f"{self.tname} must have _dummy strategy as its only consumer"
            return super().add_consumer(consumer)
        
        def _is_dummy_strategy(self):
            if self._consumers:
                # NOTE: dummy strategy will always be the only consumer
                consumer = self._consumers[0]
                return isinstance(consumer, BaseStrategy) and consumer.name == '_dummy'
            else:
                return False
        
        def load(self):
            if self._is_signal_prepared():
                super().load()
        
        def clear_dfs(self):
            assert self.engine.mode == 'event_driven'
            if not self._is_signal_prepared():
                self._data_tool.clear_df()
            for model in self.models.values():
                model.clear_dfs()
        
        def _add_raw_df(self, data, df):
            return self._data_tool.add_raw_df(data, df)
        
        def _set_data_periods(self, datas, **kwargs):
            return self._data_tool.set_data_periods(datas, **kwargs)
        
        def _prepare_df(self):
            ts_col_type = 'timestamp' if self.engine.mode == 'event_driven' else 'datetime'
            return self._data_tool.prepare_df(ts_col_type=ts_col_type)
    
        def _append_to_df(self, **kwargs):
            if not self._is_signal_prepared() and self.engine.append_signals:
                return self._data_tool.append_to_df(self.data, self.predictions, **kwargs)
        
        def _is_signal_prepared(self):
            if self._is_dummy_strategy():
                return False
            elif self.engine.mode == 'vectorized':
                return True
            elif self.engine.mode == 'event_driven':
                return self.engine.use_trained_models
                
        def next(self, data: BaseData):
            if not self._is_signal_prepared():
                return super().next(data)
            else:
                try:
                    # FIXME: pandas specific
                    # retrieve prepared signal from self.signal
                    new_pred = self.signal.loc[(data.dt, repr(data.product), repr(data.resolution))]
                    return new_pred.to_numpy()
                except Exception as e:
                    raise Exception(f"Please make sure {self.tname} has been prepared/dumped correctly") from e
        
        # FIXME: pandas specific
        def assert_consistent_signals(self):
            '''Asserts consistent model signals from vectorized and event-driven backtesting, triggered in event-driven backtesting'''
            import pandas.testing as pdt
            event_driven_signal = self.signal
            # set signal to None and load the vectorized_signal
            self.set_signal(None)
            self.load()
            assert self.signal is not None, f"Please dump your model '{self.name}' by calling model.dump() before running event-driven backtesting"
            vectorized_signal = self.signal
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