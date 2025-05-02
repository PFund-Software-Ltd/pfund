from abc import ABCMeta


class MetaEngine(ABCMeta):
    _locked = False

    def __setattr__(cls, key, value):
        if cls._locked:
            raise AttributeError(f"{key} is already set and locked")
        super().__setattr__(key, value)

    def lock(cls):
        cls._locked = True
