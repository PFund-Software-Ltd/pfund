from pydantic import BaseModel
from pfeed.typing import tDATA_LAYER, tSTORAGE


class StorageConfig(BaseModel):
    data_layer: tDATA_LAYER='clean'
    data_domain: str=''
    from_storage: tSTORAGE | None=None
    # configs specific to the storage type, for MinIO, it's access_key and secret_key etc.
    storage_options: dict | None=None
