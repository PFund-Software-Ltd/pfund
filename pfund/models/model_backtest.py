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
        def __init__(self, *_args, **_kwargs):
            Model.__init__(self, *_args, **_kwargs)
            self.initialize_mixin()
            self.add_model_signature(*_args, **_kwargs)

        # __getattr__ at this level to get the correct model name
        def __getattr__(self, attr):
            '''gets triggered only when the attribute is not found'''
            try:
                return super().__getattr__(attr)
            except AttributeError:
                class_name = Model.__name__
                raise AttributeError(f"'{class_name}' object or '{class_name}.ml_model' or '{class_name}.data_tool' has no attribute '{attr}', make sure super().__init__() is called in your strategy {class_name}.__init__()")
        
        def to_dict(self):
            model_dict = super().to_dict()
            model_dict['class'] = Model.__name__
            model_dict['model_signature'] = self._model_signature
            model_dict['data_signatures'] = self._data_signatures
            return model_dict

        def _add_consumer_datas_if_no_data(self) -> list[BaseData]:
            consumer_datas = super()._add_consumer_datas_if_no_data()
            for data in consumer_datas:
                df = self._consumer.get_raw_df(data)
                self.add_raw_df(data, df)
            return consumer_datas
        
        def _is_prepared_signal_required(self):
            # is_dummy_strategy=True means No actual strategy, only model is running in backtesting
            is_dummy_strategy = isinstance(self._consumer, BaseStrategy) and self._consumer.name == '_dummy'
            if is_dummy_strategy:
                return False
            else:
                return (
                    self.engine.mode == 'vectorized' or \
                    (self.engine.mode == 'event_driven' and \
                        self.engine.use_prepared_signals)
                )
        
        def start(self):
            super().start()
            is_dummy_strategy = isinstance(self._consumer, BaseStrategy) and self._consumer.name == '_dummy'
            if not self._is_prepared_signal_required():
                if not is_dummy_strategy:
                    self.data_tool._clear_df()
                # make loaded signal (if any) None
                self.set_signal(None)
        
        def _prepare_df_with_models(self, *args, **kwargs):
            if self.engine.mode == 'vectorized':
                self.data_tool._prepare_df_with_models(*args, **kwargs)
                
        def _append_to_df(self, *args, **kwargs):
            if not self._is_prepared_signal_required():
                return self.data_tool._append_to_df(*args, **kwargs)
        
        def next(self):
            if not self._is_prepared_signal_required():
                return super().next()
            else:
                # FIXME: pandas specific
                # retrieve prepared signal from self.signal
                new_pred = self.signal.loc[(self.data.dt, repr(self.data.product), repr(self.data.resolution))]
                return new_pred.to_numpy()
        
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
            
                
    if not issubclass(Model, BaseFeature):
        return _BacktestModel(ml_model, *args, **kwargs)
    else:
        return _BacktestModel(*args, **kwargs)