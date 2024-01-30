import os
from pathlib import Path


PROJ_NAME = Path(__file__).resolve().parents[2].name
MAIN_PATH = Path(__file__).resolve().parents[3]
PROJ_PATH = MAIN_PATH / PROJ_NAME / PROJ_NAME
EXCHANGE_PATH = PROJ_PATH / 'exchanges'
CONFIG_PATH = PROJ_PATH / 'config'
LOG_PATH = MAIN_PATH / PROJ_NAME / 'logs'
STRATEGY_PATH = PROJ_PATH / 'strategies'
MODEL_PATH = PROJ_PATH / 'models'


# paths for storing data in user's machine
PFUND_PATH = Path.home() / '.pfund'
PFUND_TRAINED_MODEL_PATH = PFUND_PATH / 'trained_models'
for path in [PFUND_PATH, PFUND_TRAINED_MODEL_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
