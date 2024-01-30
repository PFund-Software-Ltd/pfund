import os
import logging
import logging.config
from pathlib import Path

from typing import Literal

from pfund.const.paths import LOG_PATH, CONFIG_PATH
from pfund.logging.config import load_logging_config, LoggingDictConfigurator


def print_all_loggers():
    for name, logger in logging.Logger.manager.loggerDict.items():
        if hasattr(logger, 'handlers'):
            print(name, logger, logger.handlers)


def set_up_loggers(log_path: str | Path=LOG_PATH, config_path: str | Path=CONFIG_PATH):
    print('Setting up loggers...')
    logging_config: dict = load_logging_config(str(log_path), str(config_path))
    log_path = logging_config['log_path']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(f'created {log_path=}')
    # ≈ logging.config.dictConfig(logging_config) with a custom configurator
    LoggingDictConfigurator(logging_config).configure()
    
    
def create_dynamic_logger(name: str, type_: Literal['strategy', 'model', 'manager']):
    """Set up logger for strategy/model/manager
    
    Since loggers for strategy/model/manager require dynamic names,
    set_up_loggers() will not create loggers for strategy/model/manager.
    Instead, they will be created here and use the logger config of strategy/model/manager by default if not specified.
    """
    assert name, "logger name cannot be empty/None"
    assert type_ in ['strategy', 'model', 'manager'], f"Unsupported {type_=}"
    
    # NOTE: LoggingDictConfigurator is a singleton, since it has already configured in set_up_loggers(), no need to pass in config to it
    config = LoggingDictConfigurator(config=None)
    
    logging_config = config.config_orig
    loggers_config = logging_config['loggers']
    logger_name = name.lower()
    if logger_name in loggers_config:
        logger_config = loggers_config[logger_name]
    else:
        # use strategy*/model*/manager* config as default if not specified
        logger_config = loggers_config[f'_{type_}']
    if type_ not in logger_name:
        logger_name += f'_{type_}'
    
    config.configure_logger(logger_name, logger_config)
    return logging.getLogger(logger_name.lower())
