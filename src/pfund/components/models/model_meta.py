# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportFunctionMemberAccess=false, reportUnknownArgumentType=false
from __future__ import annotations

from typing import Any, Callable, Literal

from abc import ABCMeta


class MetaModel(ABCMeta):
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
    ) -> MetaModel:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        module_name = namespace.get("__module__", "")
        if MetaModel._is_user_defined_class(module_name):
            # capture the class's OWN __init__ from its namespace, NOT cls.__init__:
            # cls.__init__ can be inherited (e.g. the parent's wrapper in a user
            # subclass chain), which would skip this class's init and run the parent's twice
            original_init: Callable[..., None] | None = namespace.get("__init__")
            # during Ray's pickling the class is re-created with a namespace whose
            # __init__ is already the wrapper below — never wrap a wrapper
            is_already_wrapped = getattr(original_init, "__pfund_init_wrapper__", False)
            if original_init is not None and not is_already_wrapped:

                def init_in_correct_order(self: Any, *args: Any, **kwargs: Any):
                    # force to init the BaseClass first, e.g. SKLearnModel
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
        module_name: str = namespace.get("__module__", "")
        if MetaModel._is_user_defined_class(module_name):
            # this class's OWN init stored by __new__; None when the class defines no
            # __init__ (nothing to validate — the inherited init was already validated)
            original_init: Callable[..., None] | None = cls.__dict__.get(
                "__original_init__"
            )
            if original_init is not None:
                from pfund_kit.utils.function import get_function_args_and_kwargs

                init_args, _, _, _ = get_function_args_and_kwargs(original_init)
                # assert users to include 'model' as the first argument in __init__()
                MetaModel._assert_required_arg(cls, init_args)

        if name == "_BacktestModel":
            assert "__init__" not in namespace, (
                "In order to keep the MRO clean, _BacktestModel is not allowed to have __init__()"
            )

    @staticmethod
    def _get_required_arg() -> Literal["model"]:
        return "model"

    @classmethod
    def _assert_required_arg(mcs, cls: type, init_args: list[str]) -> None:
        required_arg = MetaModel._get_required_arg()
        if required_arg not in init_args or init_args[0] != required_arg:
            raise TypeError(
                f"{cls.__name__}.__init__() must include `{required_arg}` as the first argument after `self`, like this:\n"
                + f"{cls.__name__}.__init__(self, {required_arg}, *args, **kwargs)"
            )
