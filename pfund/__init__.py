import sys
from importlib.metadata import version

from pfund.config_handler import configure
from pfund.const.aliases import ALIASES
from pfund.const.paths import PROJ_PATH
# add python path so that for files like "ibapi" (official python code from IB) can find their modules
sys.path.append(f'{PROJ_PATH}/externals')


# NOTE: dynamically import modules to avoid click cli latency (reduced from ~4s to ~0.2s)
def __getattr__(name):
    """
    Dynamically import and return modules and classes based on their name.
    
    Supports dynamic loading of data sources and feed classes to minimize
    initial load time.
    """
    import importlib
    if 'Engine' in name:
        Engine = getattr(importlib.import_module('pfund.engines'), name)
        globals()[name] = Engine
        return Engine
    elif 'Strategy' in name:
        Strategy = getattr(importlib.import_module('pfund.strategies'), name)
        globals()[name] = Strategy
        return Strategy
    elif 'Model' in name or 'Feature' in name:
        Model = getattr(importlib.import_module('pfund.models'), name)
        globals()[name] = Model
        return Model
    elif 'Indicator' in name:
        Indicator = getattr(importlib.import_module('pfund.indicators'), name)
        globals()[name] = Indicator
        return Indicator


# NOTE: dummy classes/modules for type hinting
# e.g. import pfund as pf, when you type "pf.", 
# you will still see the following suggestions even they are dynamically imported:
BacktestEngine: ...
TradeEngine: ...
TrainEngine: ...
SandboxEngine: ...
Strategy: ...
Model: ...
PytorchModel: ...
SklearnModel: ...
Feature: ...
TaIndicator: ...
TalibIndicator: ...


__version__ = version('pfund')
__all__ = (
    '__version__',
    'configure', 
    'ALIASES',
    'BacktestEngine', 
    'TradeEngine',
    'TrainEngine', 
    'SandboxEngine',
    'Strategy', 
    'Model', 
    'PytorchModel', 
    'SklearnModel',
    'Feature', 
    'TaIndicator', 
    'TalibIndicator',
)