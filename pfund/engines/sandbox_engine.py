from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.common_literals import tSUPPORTED_DATA_TOOLS
    from pfund.config_handler import ConfigHandler

from pfund.engines.trade_engine import TradeEngine


class SandboxEngine(TradeEngine):
    def __new__(
        cls, 
        *, 
        data_tool: tSUPPORTED_DATA_TOOLS='pandas', 
        df_max_rows: int=1000,
        zmq_port=5557,
        config: ConfigHandler | None=None, 
        **settings
    ):
        return super().__new__(
            cls, 
            env='SANDBOX', 
            data_tool=data_tool,
            df_max_rows=df_max_rows,
            zmq_port=zmq_port, 
            config=config, 
            **settings
        )
    
    def __init__(
        self, 
        *, 
        data_tool: tSUPPORTED_DATA_TOOLS='pandas', 
        df_max_rows: int=1000,
        zmq_port=5557, 
        config: ConfigHandler | None=None,
        **settings
    ):  
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            super().__init__(
                env='SANDBOX', 
                data_tool=data_tool,
                df_max_rows=df_max_rows,
                zmq_port=zmq_port,
                config=config,
                **settings
            )
