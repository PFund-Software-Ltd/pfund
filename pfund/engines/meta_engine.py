from abc import ABCMeta
from collections import defaultdict


class MetaEngine(ABCMeta):
    _locked_classes = defaultdict(lambda: False)
    _first_engine_cls = None  # Tracks the first concrete engine class created
    _attrs_not_locked = ('_num',)

    def __call__(cls, *args, **kwargs):
        # Prevent BaseEngine from being instantiated
        if cls.__name__ == "BaseEngine":
            raise TypeError("BaseEngine cannot be instantiated directly")

        # First engine class allowed
        if MetaEngine._first_engine_cls is None:
            MetaEngine._first_engine_cls = cls
        # If another engine class tries to instantiate
        elif cls is not MetaEngine._first_engine_cls:
            raise RuntimeError(
                f"{cls.__name__} cannot be instantiated because "
                f"{MetaEngine._first_engine_cls.__name__} is already in use"
            )
        return super().__call__(*args, **kwargs)

    def __setattr__(cls, key, value):
        if MetaEngine._locked_classes[cls] and key not in MetaEngine._attrs_not_locked:
            raise AttributeError(f"{key} is already set and locked for class {cls.__name__}")
        super().__setattr__(key, value)

    def lock(cls):
        MetaEngine._locked_classes[cls] = True