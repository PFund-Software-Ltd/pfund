import pytest
import pfund as pf


def test_import_backtest_engine():
    try:
        from pfund import BacktestEngine
    except ImportError:
        pytest.fail("Failed to import 'BacktestEngine' package")

def test_import_trade_engine():
    try:
        from pfund import TradeEngine
    except ImportError:
        pytest.fail("Failed to import 'TradeEngine' package")

def test_import_train_engine():
    from pfund import TrainEngine
    assert TrainEngine is not None, "Failed to import 'TrainEngine'"

def test_import_sandbox_engine():
    try:
        from pfund import SandboxEngine
    except ImportError:
        pytest.fail("Failed to import 'SandboxEngine' package")
            
def test_import_strategy():
    try:
        from pfund import Strategy
    except ImportError:
        pytest.fail("Failed to import 'Strategy' package")

def test_import_model():
    try:
        from pfund import Model
    except ImportError:
        pytest.fail("Failed to import 'Model' package")

def test_import_pytorch_model():
    try:
        from pfund import PytorchModel
    except ImportError:
        pytest.fail("Failed to import 'PytorchModel' package")

def test_import_sklearn_model():
    try:
        from pfund import SklearnModel
    except ImportError:
        pytest.fail("Failed to import 'SklearnModel' package")

def test_import_feature():
    try:
        from pfund import Feature
    except ImportError:
        pytest.fail("Failed to import 'Feature' package")

def test_import_ta_indicator():
    try:
        from pfund import TAIndicator
    except ImportError:
        pytest.fail("Failed to import 'TAIndicator' package")

def test_import_talib_indicator():
    try:
        from pfund import TALibIndicator
    except ImportError:
        pytest.fail("Failed to import 'TALibIndicator' package")

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