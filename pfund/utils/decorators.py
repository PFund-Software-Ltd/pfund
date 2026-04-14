from typing import TypeVar, Callable

F = TypeVar('F', bound=Callable[..., object])


def ray_method(func: F) -> F:
    """Marker decorator indicating that a seemingly redundant method exists
    because Ray actors only support method calls, not direct attribute access."""
    return func
