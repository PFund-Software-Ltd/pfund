import pytest
import pfund as pf


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

@pytest.mark.smoke
def test_import_all():
    for attr in pf.__all__:
        assert hasattr(pf, attr), f"Package 'pfund' does not have '{attr}'"

@pytest.mark.smoke
def test_import_ibapi():
    try:
        import ibapi
    except ImportError:
        pytest.fail("Failed to import 'ibapi' package")