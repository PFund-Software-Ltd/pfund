from __future__ import annotations
from typing import TYPE_CHECKING, Literal, cast
if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.engines.engine_context import DataRangeDict
    from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings
    
from pfund.enums import Environment
from pfund.engines.trade_engine import TradeEngine


class SandboxEngine(TradeEngine):
    def __init__(
        self,
        name: str='engine',
        data_range: str | Resolution | DataRangeDict | Literal['ytd']='ytd',
        settings: SandboxEngineSettings | None=None,
    ):
        super().__init__(
            env=Environment.SANDBOX, 
            name=name, 
            data_range=data_range, 
            settings=settings,
        )
    
    @property
    def settings(self) -> SandboxEngineSettings:
        return cast("SandboxEngineSettings", self._context.settings)
    