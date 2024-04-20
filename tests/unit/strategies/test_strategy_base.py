import pytest
from pathlib import Path

from pfund.strategies.strategy_base import BaseStrategy as Strategy


@pytest.mark.smoke
def test_init():
    args = (1, 2, 3)
    kwargs = {'a': 1, 'b': 2, 'c': 3}
    strategy = Strategy(*args, **kwargs)
    assert strategy._args == args
    assert strategy._kwargs == kwargs
    
def test_load_config_direct_input():
    config = {'key': 'value'}
    Strategy.load_config(config)
    assert Strategy.config == config, "Config should match the provided dictionary"

def test_load_params_direct_input():
    params = {'param1': 10, 'param2': 20}
    strategy = Strategy()
    strategy.load_params(params)
    assert strategy.params == params, "Params should match the provided dictionary"
    
def test_load_config_from_file(mocker):
    Strategy._file_path = Path("/fake/path")
    expected_config = {'key': 'value'}
    mock_load_yaml_file = mocker.patch(
        'pfund.strategies.strategy_base.load_yaml_file', 
        return_value=expected_config
    )
    Strategy.load_config()
    assert Strategy.config == expected_config, "Config should be loaded from file"
    mock_load_yaml_file.assert_called()

def test_load_params_from_file(mocker):
    strategy = Strategy()
    strategy._file_path = Path("/fake/path")
    expected_params = {'param1': 10, 'param2': 20}
    mock_load_yaml_file = mocker.patch(
        'pfund.strategies.strategy_base.load_yaml_file', 
        return_value=expected_params
    )
    strategy.load_params()
    assert strategy.params == expected_params, "Params should be loaded from file"
    mock_load_yaml_file.assert_called()
