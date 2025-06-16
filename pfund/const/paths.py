from pathlib import Path
from platformdirs import user_log_dir, user_data_dir, user_config_dir, user_cache_dir


# project paths
PROJ_NAME = Path(__file__).resolve().parents[1].name
MAIN_PATH = Path(__file__).resolve().parents[2]
PROJ_PATH = MAIN_PATH / PROJ_NAME

# user paths
LOG_PATH = Path(user_log_dir()) / PROJ_NAME
CONFIG_PATH = Path(user_config_dir()) / PROJ_NAME / 'config'
CONFIG_FILENAME = f'{PROJ_NAME}_config.yml'
CONFIG_FILE_PATH = CONFIG_PATH / CONFIG_FILENAME
CACHE_PATH = Path(user_cache_dir()) / PROJ_NAME
DATA_PATH = Path(user_data_dir()) / PROJ_NAME
