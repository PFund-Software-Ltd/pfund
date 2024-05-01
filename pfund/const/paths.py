from pathlib import Path
from platformdirs import user_log_dir, user_data_dir, user_config_dir


# project paths
PROJ_NAME = Path(__file__).resolve().parents[1].name
MAIN_PATH = Path(__file__).resolve().parents[2]
PROJ_PATH = MAIN_PATH / PROJ_NAME

EXCHANGE_PATH = PROJ_PATH / 'exchanges'
PROJ_CONFIG_PATH = PROJ_PATH / 'config'


# user paths
LOG_PATH = Path(user_log_dir()) / PROJ_NAME
USER_CONFIG_PATH = Path(user_config_dir()) / PROJ_NAME
USER_CONFIG_FILE_PATH = USER_CONFIG_PATH / f'{PROJ_NAME}_config.yml'
DATA_PATH = Path(user_data_dir()) / PROJ_NAME
STRATEGY_PATH = DATA_PATH / 'hub' / 'strategies'
MODEL_PATH = DATA_PATH / 'hub' / 'models'
FEATURE_PATH = DATA_PATH / 'hub' / 'features'
INDICATOR_PATH = DATA_PATH / 'hub' / 'indicators'
BACKTEST_PATH = DATA_PATH / 'backtests'
NOTEBOOK_PATH = DATA_PATH / 'templates' / 'notebooks'
SPREADSHEET_PATH = DATA_PATH / 'templates' / 'spreadsheets'
DASHBOARD_PATH = DATA_PATH / 'templates' / 'dashboards'
ARTIFACT_PATH = DATA_PATH / 'artifacts'
