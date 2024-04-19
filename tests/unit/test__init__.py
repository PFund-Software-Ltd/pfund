import importlib

def test_import_backtest_engine():
    from pfund import BacktestEngine
    assert BacktestEngine is not None, "Failed to import 'BacktestEngine'"

def test_import_trade_engine():
    from pfund import TradeEngine
    assert TradeEngine is not None, "Failed to import 'TradeEngine'"

def test_import_train_engine():
    from pfund import TrainEngine
    assert TrainEngine is not None, "Failed to import 'TrainEngine'"

def test_import_test_engine():
    from pfund import SandboxEngine
    assert SandboxEngine is not None, "Failed to import 'SandboxEngine'"
            
def test_import_strategy():
    from pfund import Strategy
    assert Strategy is not None, "Failed to import 'Strategy'"

def test_import_model():
    from pfund import Model
    assert Model is not None, "Failed to import 'Model'"

def test_import_pytorch_model():
    from pfund import PyTorchModel
    assert PyTorchModel is not None, "Failed to import 'PyTorchModel'"

def test_import_sklearn_model():
    from pfund import SKLearnModel
    assert SKLearnModel is not None, "Failed to import 'SKLearnModel'"

def test_import_feature():
    from pfund import Feature
    assert Feature is not None, "Failed to import 'Feature'"

def test_import_ta_indicator():
    from pfund import TAIndicator
    assert TAIndicator is not None, "Failed to import 'TAIndicator'"

def test_import_talib_indicator():
    from pfund import TALibIndicator
    assert TALibIndicator is not None, "Failed to import 'TALibIndicator'"
    
def test_import_entire_package():
    import pfund as pf
    assert pf is not None, "Failed to import 'pfund' package"
    assert hasattr(pf, 'BacktestEngine'), "Package 'pfund' does not have 'BacktestEngine'"
    assert hasattr(pf, 'TradeEngine'), "Package 'pfund' does not have 'TradeEngine'"
    assert hasattr(pf, 'TrainEngine'), "Package 'pfund' does not have 'TrainEngine'"
    assert hasattr(pf, 'SandboxEngine'), "Package 'pfund' does not have 'SandboxEngine'"
    assert hasattr(pf, 'Strategy'), "Package 'pfund' does not have 'Strategy'"
    assert hasattr(pf, 'Model'), "Package 'pfund' does not have 'Model'"
    assert hasattr(pf, 'PyTorchModel'), "Package 'pfund' does not have 'PyTorchModel'"
    assert hasattr(pf, 'SKLearnModel'), "Package 'pfund' does not have 'SKLearnModel'"
    assert hasattr(pf, 'Feature'), "Package 'pfund' does not have 'Feature'"
    assert hasattr(pf, 'TAIndicator'), "Package 'pfund' does not have 'TAIndicator'"
    assert hasattr(pf, 'TALibIndicator'), "Package 'pfund' does not have 'TALibIndicator'"

def test_dynamic_import_backtest_engine():
    from pfund import __getattr__
    importlib.import_module('pfund.engines')
    BacktestEngine = __getattr__('BacktestEngine')
    assert BacktestEngine is not None, "Failed to dynamically import 'BacktestEngine'"

def test_dynamic_import_trade_engine():
    from pfund import __getattr__
    importlib.import_module('pfund.engines')
    TradeEngine = __getattr__('TradeEngine')
    assert TradeEngine is not None, "Failed to dynamically import 'TradeEngine'"

def test_dynamic_import_train_engine():
    from pfund import __getattr__
    importlib.import_module('pfund.engines')
    TrainEngine = __getattr__('TrainEngine')
    assert TrainEngine is not None, "Failed to dynamically import 'TrainEngine'"

def test_dynamic_import_test_engine():
    from pfund import __getattr__
    importlib.import_module('pfund.engines')
    SandboxEngine = __getattr__('SandboxEngine')
    assert SandboxEngine is not None, "Failed to dynamically import 'SandboxEngine'"

def test_dynamic_import_strategy():
    from pfund import __getattr__
    importlib.import_module('pfund.strategies')
    Strategy = __getattr__('Strategy')
    assert Strategy is not None, "Failed to dynamically import 'Strategy'"

def test_dynamic_import_model():
    from pfund import __getattr__
    importlib.import_module('pfund.models')
    Model = __getattr__('Model')
    assert Model is not None, "Failed to dynamically import 'Model'"

def test_dynamic_import_pytorch_model():
    from pfund import __getattr__
    importlib.import_module('pfund.models')
    PyTorchModel = __getattr__('PyTorchModel')
    assert PyTorchModel is not None, "Failed to dynamically import 'PyTorchModel'"

def test_dynamic_import_sklearn_model():
    from pfund import __getattr__
    importlib.import_module('pfund.models')
    SKLearnModel = __getattr__('SKLearnModel')
    assert SKLearnModel is not None, "Failed to dynamically import 'SKLearnModel'"

def test_dynamic_import_feature():
    from pfund import __getattr__
    importlib.import_module('pfund.models')
    Feature = __getattr__('Feature')
    assert Feature is not None, "Failed to dynamically import 'Feature'"

def test_dynamic_import_ta_indicator():
    from pfund import __getattr__
    importlib.import_module('pfund.indicators')
    TAIndicator = __getattr__('TAIndicator')
    assert TAIndicator is not None, "Failed to dynamically import 'TAIndicator'"

def test_dynamic_import_talib_indicator():
    from pfund import __getattr__
    importlib.import_module('pfund.indicators')
    TALibIndicator = __getattr__('TALibIndicator')
    assert TALibIndicator is not None, "Failed to dynamically import 'TALibIndicator'"
