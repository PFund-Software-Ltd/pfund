from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pfeed_pfund.component.data_handler import PFundComponentDataHandler
    from pfeed_pfund.component.data_model import PFundComponentDataModel

from pfeed.io.base_io import BaseIO
from pydantic import Field

from pfeed_pfund.component.requests.base_request import PFundComponentFeedBaseRequest


class PFundComponentFeedRetrieveRequest(PFundComponentFeedBaseRequest):
    extract_type: ClassVar[str] = "retrieve"
    io_for_retrieval: BaseIO = Field(
        description="IO used for data retrieval, not for loading data to storage"
    )

    def should_clean_data(self) -> bool:
        # artifacts are returned as stored
        return False

    def create_handler(self, data_model: PFundComponentDataModel | None = None) -> PFundComponentDataHandler:
        """Handler reading this request's artifact from storage.

        Args:
            data_model: the artifact to read, defaults to the request's.
        """
        from pfeed.enums import DataCategory

        data_model = data_model or self.to_data_model()
        return data_model.DataHandler(
            data_model=data_model,
            io=self.io_for_retrieval,
            data_layer=self.data_layer,
            data_domain=str(DataCategory.COMPONENT_DATA),
        )
