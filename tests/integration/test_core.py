import pytest

import pfund as pf
import ta
from talib import abstract as talib
import torch.nn as nn
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


class PytorchLinearRegression(nn.Module):
    '''Linear Regression for testing'''
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
        
    def forward(self, x):
        return self.linear(x)


class TestPytorch(pf.PytorchModel):
    def predict(self, *args, **kwargs):
        pass
    

class TestSklearn(pf.SklearnModel):
    def predict(self, *args, **kwargs):
        pass
    


@pytest.mark.smoke
class TestCore:
    def test_vectorized_backtesting_flow(self, mocker):
        FakeVectorizedStrategy = type('FakeVectorizedStrategy', (pf.Strategy,), {})
        engine = pf.BacktestEngine(mode='vectorized')
        strategy = engine.add_strategy(FakeVectorizedStrategy(), name='fake_vectorized_strategy', is_parallel=False)
        
        # TODO: move to conftest.py as fixture
        mock_get_historical_data = mocker.patch.object(strategy, 'get_historical_data')
        mock_data_tool = mocker.patch.object(strategy, 'data_tool')
        mock_engine_run = mocker.patch.object(engine, 'run')
        
        yf_datas = strategy.add_data(
            'IB', 'AAPL', 'USD', 'STK', resolutions=['1d'],
            backtest={
                'data_source': 'YAHOO_FINANCE',
                'start_date': '2024-01-01',
                'end_date': '2024-02-01',
            }
        )
        bybit_datas = strategy.add_data(
            'BYBIT', 'BTC', 'USDT', 'PERP', resolution='1m',
            backtest={
                'start_date': '2024-03-01',
                'end_date': '2024-03-01',
            }
        )
        
        # add pytorch model:
        strategy.add_model(TestPytorch(ml_model=PytorchLinearRegression()), name='test_pytorch', model_path='')
        
        # add sklearn model:
        strategy.add_model(TestSklearn(ml_model=SklearnLinearRegression()), name='test_sklearn', model_path='')

        # add ta indicators
        ## type 1: ta class, e.g. ta.volatility.BollingerBands
        indicator = lambda df: ta.volatility.BollingerBands(close=df['close'], window=3, window_dev=2)
        funcs = ['bollinger_mavg', 'bollinger_hband', 'bollinger_lband']
        strategy.add_indicator(pf.TAIndicator(indicator, funcs=funcs), name='BollingerBands', indicator_path='')
        ## type 2: ta function, e.g. ta.volatility.bollinger_mavg
        indicator2 = lambda df: ta.volatility.bollinger_mavg(close=df['close'], window=3)
        strategy.add_indicator(pf.TAIndicator(indicator2), name='BollingerBands2', indicator_path='')
        
        # add talib indicator
        strategy.add_indicator(pf.TALibIndicator(talib.SMA, timeperiod=3, price='close'), name='SMA', indicator_path='')
        
        engine.run()
        
        # check if datas are added to strategy correctly
        strategy_datas = [data for data_per_resolution in strategy.datas.values() for data in data_per_resolution.values()]
        assert yf_datas + bybit_datas == strategy_datas