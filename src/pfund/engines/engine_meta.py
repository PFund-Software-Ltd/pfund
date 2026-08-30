from typing import Any

from pfund_kit.utils.singleton import SingletonMeta


class EngineMeta(SingletonMeta):
    """Allows one engine per process, of one kind.

    The `pfund` logger is process-global and is configured with a single log
    path, so a second engine silently redirects the first engine's logs into
    its own file. Other process-wide state has the same problem: zmq ports,
    pfeed's auto-created data engine, and the component registry's name claims.

    Rejecting in the metaclass rather than in `BaseEngine.__init__` means the
    engine is never partially built: nothing is logged, reconfigured, or
    written to settings.toml before the error is raised.
    """

    def __call__(cls, *args: Any, **kwargs: Any):
        # Same class is fine: SingletonMeta returns the cached instance, and
        # rebuilds it in a notebook where re-running a cell is expected to.
        other = next((klass for klass in cls._instances if klass is not cls), None)
        if other is not None:
            raise RuntimeError(
                f"{other.__name__} already exists in this process; "
                + f"run {cls.__name__} in a separate process instead "
                + "(restart the kernel if you are in a notebook)"
            )
        return super().__call__(*args, **kwargs)
