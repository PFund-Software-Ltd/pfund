import pytest
import ta

import pfund as pf


yf_data_params = [
    {
        'product': 'IB_AAPL_USD_STK', 'resolutions': ['1d'],
        'start_date': '2024-01-01', 'end_date': '2024-02-01',
    },
]
    

@pytest.mark.smoke
@pytest.mark.parametrize(
    'yf_data', yf_data_params, indirect=True
)
class TestTaIndicator:
    '''Test TaIndicator as first class citizen, i.e. no strategy is added, only indicators'''
    def test_vectorized_backtesting(self, yf_data):
        engine = pf.BacktestEngine(mode='vectorized')
        # for type_ in ['type1', 'type2']:
        # FIXME
        for type_ in ['type1']:
            # type 1: ta class, e.g. ta.volatility.BollingerBands
            if type_ == 'type1':
                ta_ind = lambda df: ta.volatility.BollingerBands(close=df['close'], window=3, window_dev=2)
                funcs = ['bollinger_mavg', 'bollinger_hband', 'bollinger_lband']
                indicator = engine.add_indicator(pf.TaIndicator(ta_ind, funcs=funcs), name='BollingerBands')
            # type 2: ta function, e.g. ta.volatility.bollinger_mavg
            elif type_ == 'type2':
                ta_ind = lambda df: ta.volatility.bollinger_mavg(close=df['close'], window=3)
                indicator = engine.add_indicator(pf.TaIndicator(ta_ind), name='BollingerBands')
            
            data = indicator.add_data(
                *yf_data['product'].split('_'), resolutions=yf_data['resolutions'],
                backtest={
                    'data_source': 'YAHOO_FINANCE',
                    'start_date': yf_data['start_date'],
                    'end_date': yf_data['end_date'],
                }
            )
            
            # engine.run()