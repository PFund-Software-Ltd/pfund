from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pfund._pfeed_source.component.data_handler import PFundComponentDataHandler
    from pfund._pfeed_source.component.data_model import PFundComponentDataModel

from pfeed.io.base_io import BaseIO
from pfeed.io.blob_io import BlobIO
from pydantic import Field

from pfund._pfeed_source.component.requests_DEPRECATED.base_request import (
    PFundComponentFeedBaseRequest,
)
from pfund._pfeed_source.data_category import PFundDataCategory


class PFundComponentFeedRetrieveRequest(PFundComponentFeedBaseRequest):
    extract_type: ClassVar[str] = "retrieve"
    io_for_retrieval: BaseIO | BlobIO = Field(
        description="IO used for data retrieval, not for loading data to storage"
    )

    def create_handler(
        self, data_model: PFundComponentDataModel | None = None
    ) -> PFundComponentDataHandler:
        """Handler reading this request's artifact from storage.

        Args:
            data_model: the artifact to read, defaults to the request's.
        """
        data_model = data_model or self.to_data_model()
        return data_model.DataHandler(
            data_model=data_model,
            io=self.io_for_retrieval,
            data_layer=self.data_layer,
            data_domain=str(PFundDataCategory.COMPONENT_DATA),
        )
