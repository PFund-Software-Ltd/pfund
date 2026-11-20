from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund._pfeed_source.engine.data_model import PFundEngineDataModel  # noqa: F401, used as a string forward reference in the class bases

from pfeed.base.data_handler import BaseDataMetadata

from pfund._pfeed_source.base.data_handler import PFundBaseDataHandler


class PFundEngineDataMetadata(BaseDataMetadata):
    pass


# TODO: implement _create_dataset_key(), _validate_schema(), write_batch() and read() when the engine feed has verbs
class PFundEngineDataHandler(
    PFundBaseDataHandler["PFundEngineDataModel", PFundEngineDataMetadata]
):
    Metadata = PFundEngineDataMetadata
