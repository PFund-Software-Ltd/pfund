import os
import logging

from typing import Literal

from pfund.plogging.config import LoggingDictConfigurator
from pfund.utils.utils import load_yaml_file, get_engine_class


def print_all_loggers():
    for name, logger in logging.Logger.manager.loggerDict.items():
        if hasattr(logger, 'handlers'):
            print(name, logger, logger.handlers)
            

def set_up_loggers(log_path, logging_config_file_path, user_logging_config: dict | None=None) -> LoggingDictConfigurator:
    def deep_update(default_dict, override_dict, raise_if_key_not_exist=False):
        '''Updates a default dictionary with an override dictionary, supports nested dictionaries.'''
        for key, value in override_dict.items():
            
            # make sure log level is in uppercase, 'debug' -> 'DEBUG'
            if key == 'level':
                value = value.upper()
                
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
    # print('Setting up loggers...')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(f'created {str(log_path)}')
    logging_config: dict = load_yaml_file(logging_config_file_path)
    if user_logging_config:
        deep_update(logging_config, user_logging_config)
    logging_config['log_path'] = log_path
    # ≈ logging.config.dictConfig(logging_config) with a custom configurator
    
    logging_configurator = LoggingDictConfigurator(logging_config)
    logging_configurator.configure()
    return logging_configurator


def create_dynamic_logger(name: str, type_: Literal['strategy', 'model', 'indicator', 'feature', 'manager']):
    """Set up logger for strategy/model/manager
    
    Since loggers for strategy/model/manager require dynamic names,
    set_up_loggers() will not create loggers for strategy/model/manager.
    Instead, they will be created here and use the logger config of strategy/model/manager by default if not specified.
    """
    assert name, "logger name cannot be empty/None"
    assert type_ in ['strategy', 'model', 'indicator', 'feature', 'manager'], f"Unsupported {type_=}"
    
    Engine = get_engine_class()
    config = Engine.logging_configurator
    
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
