# NOTE: need this to make TYPE_CHECKING work to avoid the circular import issue
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.models.model_base import MachineLearningModel
    from pfund.datas.data_base import BaseData
    from pfund.types.core import tModel

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

        def add_consumer_datas_if_no_data(self) -> list[BaseData]:
            consumer_datas = super().add_consumer_datas_if_no_data()
            for data in consumer_datas:
                consumer_data_tool = self._consumer.data_tool
                df = consumer_data_tool.get_raw_df(data)
                self._data_tool.add_raw_df(data, df)
            return consumer_datas
        
        def _is_dummy_strategy(self):
            return isinstance(self._consumer, BaseStrategy) and self._consumer.name == '_dummy'
        
        def start(self):
            super().start()
            if self.engine.mode == 'event_driven':
                self._data_tool.prepare_df_before_event_driven_backtesting()
                
        def stop(self):
            super().stop()
            if self.engine.mode == 'event_driven' and self._is_dummy_strategy():
                self._assert_consistent_signals()
       
        def load(self):
            if self.engine.load_models:
                super().load()
        
        def get_df_iterable(self):
            return self._data_tool.get_df_iterable()
        
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
        
        def _prepare_df_with_signals(self):
            if self.engine.mode == 'vectorized':
                self._data_tool.prepare_df_with_signals(self.models)
        
        def _append_to_df(self, **kwargs):
            if not self._is_signal_prepared() and self.engine.append_signals:
                return self._data_tool.append_to_df(self.data, self.predictions, **kwargs)
        
        def _is_signal_prepared(self):
            if self._is_dummy_strategy():
                return False
            elif self.engine.mode == 'vectorized':
                return True
            elif self.engine.mode == 'event_driven':
                return self.engine.load_models
                
        def next(self):
            if not self._is_signal_prepared():
                return super().next()
            else:
                # FIXME: pandas specific
                # retrieve prepared signal from self.signal
                new_pred = self.signal.loc[(self.data.dt, repr(self.data.product), repr(self.data.resolution))]
                return new_pred.to_numpy()
        
        # FIXME: pandas specific
        def _assert_consistent_signals(self):
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