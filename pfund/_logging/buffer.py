import time


class LogBuffer:
    """
    A temporary logger that buffers log messages until the real logger is ready.
    Used for remote components where ZMQ PUBHandler might not be set up yet.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._buffer = []
    
    def _log(self, level: str, message: str, *args, **kwargs):
        """Internal method to buffer log messages"""
        # Format the message if args are provided
        if args:
            try:
                message = message % args
            except (TypeError, ValueError):
                # If formatting fails, just concatenate
                message = f"{message} {' '.join(str(arg) for arg in args)}"
        
        # Store log entry with timestamp
        log_entry = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            'kwargs': kwargs
        }
        self._buffer.append(log_entry)
        # TEMP
        print('appended log entry to buffer', log_entry)
    
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message"""
        self._log('DEBUG', message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log an info message"""
        self._log('INFO', message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log a warning message"""
        self._log('WARNING', message, *args, **kwargs)
    
    def warn(self, message: str, *args, **kwargs):
        """Alias for warning"""
        self.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log an error message"""
        self._log('ERROR', message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log a critical message"""
        self._log('CRITICAL', message, *args, **kwargs)
    