# pyright: reportIncompatibleVariableOverride=false
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.engines.engine_context import DataRangeDict
    from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings

from pfund.engines.trade_engine import TradeEngine
from pfund.enums import Environment


class SandboxEngine(TradeEngine):
    settings: SandboxEngineSettings

    def __init__(
        self,
        name: str = "engine",
        data_range: str
        | Resolution
        | DataRangeDict
        | tuple[str, str]
        | Literal["ytd"] = "ytd",
        settings: SandboxEngineSettings | None = None,
    ):
        super().__init__(
            env=Environment.SANDBOX,
            name=name,
            data_range=data_range,
            settings=settings,
        )

    def _assert_env(self):
        pass

    # TODO: override feed.stream(), opt in for replay etc.
