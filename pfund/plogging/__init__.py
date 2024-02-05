import os
import logging
from pathlib import Path

from typing import Literal

from pfund.plogging.config import load_logging_config, LoggingDictConfigurator


def print_all_loggers():
    for name, logger in logging.Logger.manager.loggerDict.items():
        if hasattr(logger, 'handlers'):
            print(name, logger, logger.handlers)


def set_up_loggers(log_path: Path, logging_config_file_path: Path, user_logging_config: dict | None=None):
    def deep_update(default_dict, override_dict, raise_if_key_not_exist=False):
        '''Updates a default dictionary with an override dictionary, supports nested dictionaries.'''
        for key, value in override_dict.items():
            if raise_if_key_not_exist and key not in default_dict:
                # Raise an exception if the key from override_dict doesn't exist in default_dict
                raise KeyError(f"Key '{key}' is not supported in logging config.")
            if isinstance(value, dict):
                # Get the default_dict value for key, if not exist, create a new empty dict
                default_value = default_dict.get(key, {})
                if isinstance(default_value, dict):
                    # Recursively update the dictionary
                    deep_update(default_value, value)
                else:
                    # If the default value is not a dict, replace it with the override value
                    default_dict[key] = value
            else:
                # Update the key with the override value
                default_dict[key] = value
    print('Setting up loggers...')
    if not log_path.exists():
        os.makedirs(log_path)
        print(f'created {str(log_path)}')
    logging_config: dict = load_logging_config(logging_config_file_path)
    if user_logging_config:
        deep_update(logging_config, user_logging_config)
    logging_config['log_path'] = log_path
    # ≈ logging.config.dictConfig(logging_config) with a custom configurator
    LoggingDictConfigurator(logging_config).configure()
    

# TODO: support 'feature', 'indicator'
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