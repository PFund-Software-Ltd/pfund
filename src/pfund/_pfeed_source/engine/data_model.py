from __future__ import annotations

from typing import ClassVar

from pfund._pfeed_source.base_data_model import PFundBaseDataModel
from pfund._pfeed_source.engine.data_handler import PFundEngineDataHandler


class PFundEngineDataModel(PFundBaseDataModel):
    DataHandler: ClassVar[type[PFundEngineDataHandler]] = PFundEngineDataHandler
