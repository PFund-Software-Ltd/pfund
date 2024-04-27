import os

import pytest
import pandas as pd
import pfeed as pe

from pfund.const.paths import MAIN_PATH


def _get_data_file_path(data_source, pdt_or_symbol, resolution, rollback_period, start_date, end_date):
    data_path = MAIN_PATH / 'tests' / 'data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    data_name = f'{data_source}_{pdt_or_symbol}_{resolution}_{rollback_period}_{start_date}_{end_date}'
    data_file_path = data_path / f'{data_name}.parquet'
    return data_file_path


def _write_dfs_to_parquets_if_not_exist(request, data_source):
    data_source = data_source.upper()
    product = request.param['product']
    trading_venue, bccy, qccy, ptype = product.split('_')
    if data_source == 'YAHOO_FINANCE':
        pdt_or_symbol = bccy
    else:
        pdt_or_symbol = '_'.join([bccy, qccy, ptype])
    resolutions: list = request.param['resolutions']
    rollback_period = request.param.get('rollback_period', '1w')
    start_date = request.param.get('start_date', '')
    end_date = request.param.get('end_date', '')
    if data_source == 'YAHOO_FINANCE':
        feed = pe.YahooFinanceFeed()
    elif data_source == 'BYBIT':
        feed = pe.BybitFeed()
    # TODO: other feeds
    else:
        raise NotImplementedError
    for resolution in resolutions:
        data_file_path = _get_data_file_path(data_source, pdt_or_symbol, resolution, rollback_period, start_date, end_date)
        # if data doesn't exist, call the actual api and save the data to file
        if not os.path.exists(data_file_path):
            df = feed.get_historical_data(pdt_or_symbol, rollback_period=rollback_period, start_date=start_date, end_date=end_date, resolution=resolution)
            df.to_parquet(data_file_path)


def get_historical_data(
    pdt_or_symbol: str,
    rollback_period: str='1w',
    resolution: str='1d',
    start_date: str='',
    end_date: str='',
    data_source='',
    **kwargs
):
    assert data_source, "data_source is required"
    data_file_path = _get_data_file_path(data_source, pdt_or_symbol, resolution, rollback_period, start_date, end_date)
    df = pd.read_parquet(data_file_path)
    return df


@pytest.fixture
def yf_data(mocker, request):
    data_source = 'YAHOO_FINANCE'
    _write_dfs_to_parquets_if_not_exist(request, data_source)

    mock_YahooFinanceFeed = mocker.patch('pfeed.feeds.YahooFinanceFeed')
    mock_feed = mock_YahooFinanceFeed.return_value
    mock_feed.name = data_source
    mock_feed.get_historical_data.side_effect = lambda *args, **kwargs: get_historical_data(*args, **kwargs, data_source=data_source)
    return request.param