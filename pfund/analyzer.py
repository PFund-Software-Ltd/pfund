from __future__ import annotations
from typing import Literal
    
import os
import json
from pathlib import Path

from pfund.utils import utils
from pfund.config import get_config
from pfund.const.paths import PROJ_PATH


# TODO: cursor, windsurf, trae, zed, neovim, helix, ...
tCODE_EDITOR = Literal['vscode', 'pycharm']


class Analyzer:    
    config = get_config()
    
    notebook_path = Path(config.notebook_path)
    spreadsheet_path = Path(config.spreadsheet_path)
    dashboard_path = Path(config.dashboard_path)
    
    def __init__(self, data: dict | None=None, backtest_name=''):
        self._data = data or {}
        if backtest_name:
            self._data = self._load_backtest_history(backtest_name)
            if data:
                print('Warning: backtest_name is provided, but data is also provided, data will be ignored')
        
    def _load_backtest_history(self, backtest_name: str) -> dict:
        if '.json' not in backtest_name:
            backtest_name += '.json'
        file_path = os.path.join(self.config.backtest_path, backtest_name)
        backtest_history = {}
        with open(file_path, 'r') as f:
            backtest_history = json.load(f)
        return backtest_history
    
    @staticmethod
    def _is_file(template: str) -> bool:
        if '\\' in template or '/' in template:
            assert Path(template).resolve().is_file(), f"File {template} does not exist"
            return True
        return False
    
    @staticmethod
    def _derive_template_type(template: str) -> Literal['notebook', 'spreadsheet', 'dashboard']:
        if '.ipynb' in template:
            template_type = 'notebook'
        elif '.grid' in template:
            template_type = 'spreadsheet'
        elif '.py' in template:
            template_type = 'dashboard'
        else:
            raise ValueError(f"Template {template} is not a valid template, only .ipynb, .grid, .py are supported.")
        return template_type
    
    def _find_template(self, template: str) -> str:
        '''Check if the template exists in pfund's templates or user's templates
        e.g. template = 'notebook.ipynb' or 'spreadsheet.grid' or 'dashboard.py'
        '''
        template_type = self._derive_template_type(template)
        pfund_templates_dir = PROJ_PATH / 'templates' / (template_type+'s')
        user_templates_dir = getattr(self, f'{template_type}_path')
        for templates_dir in [pfund_templates_dir, user_templates_dir]:
            for file_name in os.listdir(templates_dir):
                if template == file_name:
                    template_file_path = templates_dir / template
                    return str(template_file_path)
        else:
            raise FileNotFoundError(f"Template {template} not found in pfund's templates or user's templates")
    
    def _get_editor_cmd(self, editor: tCODE_EDITOR) -> str:
        if editor == 'vscode':
            cmd = 'code'
            if utils.is_command_available(cmd):
                return cmd
            else:
                print("VSCode command 'code' is not available, cannot open the output notebook")
        elif editor == 'pycharm':
            for cmd in ['charm', 'pycharm']:
                if utils.is_command_available(cmd):
                    return cmd
            else:
                print("PyCharm commands 'charm'/'pycharm' are both not available, cannot open the output notebook")
        else:
            print(f"Editor '{editor}' is not supported, cannot open the output notebook")
    
    def run_notebooks(
        self, 
        notebooks: list[str] | str,
        *voila_args,
        data: dict | None=None,
        display: bool=True,
        port: int=8866,
        show_results_only: bool=True,
        open_outputs: bool=False,
        outputs_path: str | None=None,
        editor: tCODE_EDITOR='vscode'
    ) -> None:
        '''
        Args:
            notebook:
                - notebook_template's name
                - notebook's full path in str or Path
            voila_args: additional arguments to pass to voila
            data: data to be analyzed, if None, use the data passed to the Analyzer instance during initialization
            display: if True, display the notebook in voila
            show_results_only: if True, display only the results (no source code) in voila
            open_outputs: if True, open the output notebook in the editor
            outputs_path: path to save the output notebooks, if None, do not save the output notebooks
        '''
        import subprocess
        import papermill as pm
        
        def _find_available_port(_port):
            retry_num = 100
            while retry_num:
                if not utils.is_port_in_use(_port):
                    return _port
                retry_num -= 1
                _port += 1
            else:
                raise Exception(f"No available ports found starting from {_port - 100}, cannot display the notebook")
        
        def _assert_voila_args_are_valid():
            for arg in voila_args:
                if not arg.startswith('--'):
                    raise ValueError(f"Voila argument '{arg}' should start with '--'")
                if arg.startswith('--port='):
                    raise ValueError(f"Voila argument '{arg}' should not be passed in, use the 'port' argument instead")
                if arg.startswith('--strip_sources='):
                    raise ValueError(f"Voila argument '{arg}' should not be passed in, use the 'show_results_only' argument instead")
            
        voila_processes = []
        nb_output_file_paths = []
        if isinstance(notebooks, str):
            notebooks = [notebooks]
        data = data or self._data
        if not data:
            raise ValueError("No data passed in or stored in the Analyzer instance, please pass in the data to be analyzed.")
        
        if open_outputs:
            assert outputs_path is not None, f"{outputs_path=}, cannot open the output notebook without saving it."
            editor_cmd = self._get_editor_cmd(editor)
        
        _assert_voila_args_are_valid()
        is_theme_provided = any(arg.startswith('--theme=') for arg in voila_args)
        if not is_theme_provided:
            default_theme = 'dark'
            voila_args = [f'--theme={default_theme}', *voila_args]
            
        try:
            for notebook in notebooks:
                if self._is_file(notebook):
                    nb_input_file_path: str = notebook
                    notebook = Path(notebook).name  # e.g. 'notebook.ipynb'
                else:
                    nb_input_file_path: str = self._find_template(notebook)
                nb_output_file_path = Path(outputs_path or '.').resolve() / f'{notebook.replace(".ipynb", "")}_output.ipynb'
                nb_output_file_path = str(nb_output_file_path)
                nb_output_file_paths.append(nb_output_file_path)
                
                print(f"Executing notebook: {notebook}")
                pm.execute_notebook(
                    nb_input_file_path,
                    nb_output_file_path,
                    parameters=data
                )
                
                if open_outputs:
                    # e.g. code notebook_output.ipynb if using vscode
                    if editor_cmd:
                        subprocess.run([editor_cmd, nb_output_file_path])
                
                if display:
                    port = _find_available_port(port)
                    is_last_notebook = (notebook == notebooks[-1])
                    subprocess_func = subprocess.run if is_last_notebook else subprocess.Popen
                    process = subprocess_func([
                        'voila',
                        f'--port={port}',
                        f'--strip_sources={show_results_only}',
                        *voila_args,
                        nb_output_file_path
                    ])
                    voila_processes.append(process)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping the execution of the notebooks")
        except Exception:
            raise
        finally:
            if outputs_path is None:
                for nb_output_file_path in nb_output_file_paths:
                    print(f"{outputs_path=}, removing output notebook: {nb_output_file_path}")
                    os.remove(nb_output_file_path)
            for process in voila_processes:
                process.terminate()
                process.wait()
    
    # TODO:
    def run_spreadsheets(
        self, 
        spreadsheets: list[str] | str
    ):
        pass
    
    # TODO:
    def run_dashboards(
        self, 
        dashboards: list[str] | str
    ):
        pass