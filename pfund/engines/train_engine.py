from __future__ import annotations

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfeed.types.literals import tDATA_TOOL

from pfund.engines.backtest_engine import BacktestEngine


class TrainEngine(BacktestEngine):
    def __new__(
        cls, 
        *, 
        data_tool: tDATA_TOOL='pandas', 
        mode: Literal['vectorized' | 'event_driven']='vectorized', 
        **settings
    ):
        return super().__new__(
            cls,
            env='TRAIN',
            data_tool=data_tool,
            mode=mode,
            **settings
        )
    
    def __init__(
        self,
        *,
        data_tool: tDATA_TOOL='pandas',
        mode: Literal['vectorized' | 'event_driven']='vectorized',
        **settings
    ):
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            super().__init__(
                env='TRAIN', 
                data_tool=data_tool,
                mode=mode,
                **settings
            )
    
    def is_training(self):
        return True
    
    def run(self):
        pass