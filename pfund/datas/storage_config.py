from pydantic import BaseModel
from pfeed.typing import tDATA_LAYER, tSTORAGE


class StorageConfig(BaseModel):
    pfeed_use_ray: bool = True  # if use_ray in pfeed
    retrieve_per_date: bool = False  # refer to `retrieve_per_date` in pfeed's get_historical_data()
    data_layer: tDATA_LAYER | None=None
    from_storage: tSTORAGE | None=None
    to_storage: tSTORAGE='cache'
    # configs specific to "from_storage", for MinIO, it's access_key and secret_key etc.
    storage_options: dict | None=None
