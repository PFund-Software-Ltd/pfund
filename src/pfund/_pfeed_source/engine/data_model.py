from __future__ import annotations

from typing import ClassVar

from pfeed_pfund.base_data_model import PFundBaseDataModel
from pfeed_pfund.engine.data_handler import PFundEngineDataHandler


class PFundEngineDataModel(PFundBaseDataModel):
    DataHandler: ClassVar[type[PFundEngineDataHandler]] = PFundEngineDataHandler
