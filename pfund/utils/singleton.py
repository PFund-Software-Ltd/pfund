from threading import Lock


class Singleton:
    _instances = {}
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        # Double-checked locking pattern for thread-safe singleton
        # Check 1: Performance optimization - skip lock if instance already exists
        if cls not in cls._instances:
            with cls._lock:
                # Check 2: Correctness - another thread may have created instance while we waited
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]
    
    @classmethod
    def _remove_singleton(cls):
        with cls._lock:
            if cls in cls._instances:
                del cls._instances[cls]