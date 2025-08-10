from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.enums import DataTool
    from pfund.enums import Environment, RunMode
    from pfund._typing import DataParamsDict
    from pfund.engines.base_engine import BaseEngine
    from pfund.engines.base_engine_settings import BaseEngineSettings

import datetime
from dataclasses import dataclass

from pfund import get_config


config = get_config()


@dataclass(frozen=True)
class EngineProxy:
    name: str
    env: Environment
    run_mode: RunMode
    data_tool: DataTool
    data_start: datetime.date
    data_end: datetime.date
    settings: BaseEngineSettings
    logging_config: dict

    @classmethod
    def from_engine(cls, engine: BaseEngine) -> EngineProxy:
        return cls(
            name=engine.name,
            env=engine._env,
            run_mode=engine._run_mode,
            data_tool=engine._data_tool,
            data_start=engine._data_start,
            data_end=engine._data_end,
            settings=engine._settings,
            logging_config=engine._logging_config,
        )
    
    def get_data_params(self) -> DataParamsDict:
        '''Data params are used in components' data stores'''
        return {
            'data_start': self.data_start,
            'data_end': self.data_end,
            'data_tool': self.data_tool,
            'storage': config.storage,
            'storage_options': config.storage_options,
            'use_deltalake': config.use_deltalake,
        }
