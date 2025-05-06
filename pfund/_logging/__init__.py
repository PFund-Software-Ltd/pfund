import os
import logging


def print_all_loggers():
    for name, logger in logging.Logger.manager.loggerDict.items():
        if hasattr(logger, 'handlers'):
            print(name, logger, logger.handlers)
            

def setup_logging_config(log_path, logging_config_file_path, user_logging_config: dict | None=None) -> dict:
    from pfund.utils.utils import load_yaml_file
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
    return logging_config
