from enum import StrEnum


# EXTEND: ArtifactType.feature to write (expensive) computed features to storage
class ArtifactType(StrEnum):
    model = "model"  # e.g. sklearn model, pytorch model, etc.
    data = "data"  # tables, e.g. ducklake table
    source = "source"  # .py files
    checkpoint = "checkpoint"  # model checkpoint
