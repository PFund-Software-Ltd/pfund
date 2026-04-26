from enum import StrEnum


class ArtifactType(StrEnum):
    model = 'model'  # e.g. sklearn model, pytorch model, etc.
    data = 'data'  # e.g. parquet files
    source = 'source'  # .py files