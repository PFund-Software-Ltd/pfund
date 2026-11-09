from __future__ import annotations

from typing import ClassVar

from pfeed.base.data_model import BaseDataModel

from pfeed_pfund.engine.data_handler import PFundEngineDataHandler


# TODO: identify the engine run, e.g. env, project_name, run_name, like PFundComponentDataModel
class PFundEngineDataModel(BaseDataModel):
    DataHandler: ClassVar[type[PFundEngineDataHandler]] = PFundEngineDataHandler
