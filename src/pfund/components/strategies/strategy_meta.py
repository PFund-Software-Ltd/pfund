from __future__ import annotations

from typing import Any, Callable

from abc import ABCMeta


class MetaStrategy(ABCMeta):
    @staticmethod
    def _is_user_defined_class(module_name: str) -> bool:
        return not module_name.startswith("pfund.") and not module_name.startswith(
            "ray."
        )

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> MetaStrategy:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        module_name = namespace.get("__module__", "")
        if MetaStrategy._is_user_defined_class(module_name):
            # capture the class's OWN __init__ from its namespace, NOT cls.__init__:
            # cls.__init__ can be inherited (e.g. the parent's wrapper in a user
            # subclass chain), which would skip this class's init and run the parent's twice
            original_init: Callable[..., None] | None = namespace.get("__init__")
            # during Ray's pickling the class is re-created with a namespace whose
            # __init__ is already the wrapper below — never wrap a wrapper
            is_already_wrapped = getattr(original_init, "__pfund_init_wrapper__", False)
            if original_init is not None and not is_already_wrapped:

                def init_in_correct_order(self: Any, *args: Any, **kwargs: Any):
                    # force to init the BaseClass first, e.g. BaseStrategy
                    BaseClass = cls.__bases__[0]
                    # framework init FIRST, with the user's exact args
                    BaseClass.__init__(self, *args, **kwargs)
                    # user's own __init__ body SECOND, same args
                    cls.__original_init__(self, *args, **kwargs)

                init_in_correct_order.__pfund_init_wrapper__ = True
                cls.__init__ = init_in_correct_order
                # per-class store (no hasattr guard: each class must record its OWN init;
                # hasattr saw the parent's inherited attr and skipped the child's)
                cls.__original_init__ = original_init
        return cls

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(name, bases, namespace, **kwargs)

        if name == "_BacktestStrategy":
            assert "__init__" not in namespace, (
                "In order to keep the MRO clean, _BacktestStrategy is not allowed to have __init__()"
            )
