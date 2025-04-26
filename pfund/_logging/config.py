import sys
import time
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
    _MANUALLY_CONFIGURED_HANDLERS = [
        'file_handler',
        'compressed_timed_rotating_file_handler',
    ]
    
    def __init__(self, config):
        self.config_orig = copy.deepcopy(config)
        # remove handlers that are manually configured
        for h in list(config.get('handlers', {}).keys()):
            if h in self._MANUALLY_CONFIGURED_HANDLERS:
                del config['handlers'][h]
        # remove loggers that are manually configured, e.g. _strategy/_model/_manager
        for l in list(config.get('loggers', {}).keys()):
            if l.startswith('_'):
                del config['loggers'][l]
        DictConfigurator.__init__(self, config)
    
    def add_handlers(self, logger, handlers):
        """Add handlers to a logger from a list of names."""
        for h in handlers:
            try:
                if h in self._MANUALLY_CONFIGURED_HANDLERS:
                    handler = configure_file_handler(self, logger.name, h)
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


def get_fh_filename(log_path, logger_name):
    # filename = time.strftime(f'{log_path}/{logger_name}.{filename_format}.log')
    filename = f'{log_path}/{logger_name}.log'
    return filename


def configure_file_handler(configurator, logger_name, handler_name):
    logging_config = configurator.config_orig
    handler_config: dict = logging_config['handlers'][handler_name]
    formatter_name = handler_config.get('formatter', 'file')
    formatter = configurator.configure_formatter(logging_config['formatters'][formatter_name])
    logging_level = LEVELS[handler_config.get('level', 'DEBUG').lower()]
    filename = get_fh_filename(logging_config['log_path'], logger_name)
    Handler = configurator.resolve(handler_config['class'])
    fh = Handler(filename, **handler_config.get('kwargs', {}))
    fh.name = handler_name  # for convention, since logging.config also gives the handlers names
    fh.setLevel(logging_level)
    fh.setFormatter(formatter)
    return fh