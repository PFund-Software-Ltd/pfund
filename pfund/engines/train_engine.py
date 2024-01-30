from pfund.engines.backtest_engine import BacktestEngine
from pfund.data_tools.data_tool_base import DataTool
from pfund.engines.backtest_engine import BacktestMode


class TrainEngine(BacktestEngine):
    def __new__(cls, *, data_tool: DataTool='pandas', mode: BacktestMode='vectorized', **configs):
        return super().__new__(cls, env='TRAIN', data_tool=data_tool, mode=mode, **configs)
    
    def __init__(self, *, data_tool: DataTool='pandas', mode: BacktestMode='vectorized', **configs):
        super().__init__(env='TRAIN', data_tool=data_tool, mode=mode, **configs)
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            pass
    
    def is_training(self):
        return True
    
    def run(self):
        pass