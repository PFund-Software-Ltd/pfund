import sys
import copy
import logging
from logging.config import DictConfigurator
from logging.handlers import TimedRotatingFileHandler

from pfund._logging.filters import FullPathFilter
from pfund._logging.formatter import ColoredFormatter


LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}


# override logging's DictConfigurator as it doesn't pass in logger names to file handlers to create filenames
class LoggingDictConfigurator(DictConfigurator):
    MANUALLY_CONFIGURED_HANDLERS = [
        'file_handler',
        'compressed_timed_rotating_file_handler',
    ]
    
    def __init__(self, config: dict):
        self._pfund_config: dict = copy.deepcopy(config)
        
        # remove handlers that are manually configured
        handlers: list[str] = list(config.get('handlers', {}))
        for handler_name in handlers:
            if handler_name in self.MANUALLY_CONFIGURED_HANDLERS:
                del config['handlers'][handler_name]
                
        # remove loggers that are manually configured, e.g. _strategy/_model
        loggers: list[str] = list(config.get('loggers', {}))
        for logger_name in loggers:
            if logger_name.startswith('_'):
                del config['loggers'][logger_name]

        super().__init__(config)
    
    def add_handlers(self, logger, handlers):
        """Add handlers to a logger from a list of names."""
        for h in handlers:
            try:
                if h in self.MANUALLY_CONFIGURED_HANDLERS:
                    handler = self._configure_file_handler(logger.name, h)
                else:
                    handler = self.config['handlers'][h]
                if handler.name == 'stream_path_handler':
                    # add filter that defines %(shortpath)s
                    handler.addFilter(FullPathFilter())

                if isinstance(handler, logging.StreamHandler) and getattr(handler, 'stream', None) in [sys.stdout, sys.stderr]:
                    handler.setFormatter(ColoredFormatter(fmt=handler.formatter._fmt, datefmt=handler.formatter.datefmt))
                
                if isinstance(handler, TimedRotatingFileHandler):
                    if handler.shouldRollover(None):
                        handler.doRollover()
                logger.addHandler(handler)
            except Exception as e:
                raise ValueError('Unable to add handler %r' % h) from e

    def _configure_file_handler(self, logger_name: str, handler_name: str) -> logging.FileHandler:
        logging_config: dict = self._pfund_config
        handler_config: dict = logging_config['handlers'][handler_name]
        formatter_name = handler_config.get('formatter', 'file')
        formatter = self.configure_formatter(logging_config['formatters'][formatter_name])
        logging_level = LEVELS[handler_config.get('level', 'DEBUG').lower()]
        log_path = logging_config['log_path']
        # filename = time.strftime(f'{log_path}/{logger_name}.{filename_format}.log')
        filename = f'{log_path}/{logger_name}.log'
        Handler = self.resolve(handler_config['class'])
        fh = Handler(filename, **handler_config.get('kwargs', {}))
        fh.name = handler_name  # for convention, since logging.config also gives the handlers names
        fh.setLevel(logging_level)
        fh.setFormatter(formatter)
        return fh
