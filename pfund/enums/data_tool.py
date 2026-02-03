from enum import StrEnum
from pfeed.enums import DataTool as PFeedDataTool


# FIXME: this is a temporary enum for data tool, to be removed when pfund supports all data tools from pfeed
class DataTool(StrEnum):
    pandas = PFeedDataTool.pandas
    polars = PFeedDataTool.polars