from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.types.common_literals import tSUPPORTED_DATA_TOOLS
    from pfund.types.common_literals import tSUPPORTED_BACKTEST_MODES

from pfund.engines.backtest_engine import BacktestEngine
from pfund.config_handler import ConfigHandler


class TrainEngine(BacktestEngine):
    def __new__(
        cls, 
        *, 
        data_tool: tSUPPORTED_DATA_TOOLS='pandas', 
        mode: tSUPPORTED_BACKTEST_MODES='vectorized', 
        config: ConfigHandler | None=None, 
        **settings
    ):
        return super().__new__(
            cls,
            env='TRAIN',
            data_tool=data_tool,
            mode=mode,
            config=config,
            **settings
        )
    
    def __init__(
        self,
        *,
        data_tool: tSUPPORTED_DATA_TOOLS='pandas',
        mode: tSUPPORTED_BACKTEST_MODES='vectorized',
        config: ConfigHandler | None=None,
        **settings
    ):
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            super().__init__(
                env='TRAIN', 
                data_tool=data_tool,
                mode=mode,
                config=config,
                **settings
            )
    
    def is_training(self):
        return True
    
    def run(self):
        pass