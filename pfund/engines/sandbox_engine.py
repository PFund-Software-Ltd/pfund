from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.typing.literals import tDATA_TOOL

from pfund.engines.trade_engine import TradeEngine


class SandboxEngine(TradeEngine):
    def __new__(
        cls, 
        *, 
        data_tool: tDATA_TOOL='pandas', 
        df_min_rows: int=1_000,
        df_max_rows: int=3_000,
        zmq_port=5557,
        **settings
    ):
        return super().__new__(
            cls,
            env='SANDBOX',
            data_tool=data_tool,
            df_min_rows=df_min_rows,
            df_max_rows=df_max_rows,
            zmq_port=zmq_port,
            **settings
        )
    
    def __init__(
        self, 
        *, 
        data_tool: tDATA_TOOL='pandas', 
        df_min_rows: int=1_000,
        df_max_rows: int=3_000,
        zmq_port=5557, 
        **settings
    ):  
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            super().__init__(
                env='SANDBOX',
                data_tool=data_tool,
                df_min_rows=df_min_rows,
                df_max_rows=df_max_rows,
                zmq_port=zmq_port,
                **settings
            )
