import os
from pathlib import Path

from typing import Literal

from pfund.utils import utils
from pfund.config_handler import ConfigHandler
from pfund.const.paths import PROJ_PATH


class Analyzer:    
    try:
        Engine = utils.get_engine_class()
        config = Engine.config
    except:
        config = ConfigHandler.load_config()
    
    notebook_path = Path(config.notebook_path)
    spreadsheet_path = Path(config.spreadsheet_path)
    dashboard_path = Path(config.dashboard_path)
    
    def __init__(self, data: dict | None=None):
        self.data = data or {}
        
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
    
    def run_notebooks(
        self, 
        notebooks: list[str] | str,
        display=True,
        data: dict | None=None,
        show_results_only=True,
        open_output=False,
        save_output=False,
        output_path: str | None=None,
        editor=Literal['vscode', 'pycharm']
    ) -> list[str]:
        '''
        Args:
            notebook: 
                - notebook_template's name
                - notebook's full path in str or Path
            display: if True, display the notebook in voila
            data: data to be analyzed, if None, use the data passed to the Analyzer instance during initialization
            show_results_only: if True, display only the results (no source code) in voila
            open_output: if True, open the output notebook in the editor
            save_output: if True, save the output notebook
            output_path: path to save the output notebooks
        '''
        import subprocess
        import papermill as pm
        
        nb_output_file_paths = []
        if isinstance(notebooks, str):
            notebooks = [notebooks]
        data = data or self.data
        if not data:
            raise ValueError("No data passed in or stored in the Analyzer instance, please pass in the data to be analyzed.")
        
        for notebook in notebooks:
            if self._is_file(notebook):
                nb_input_file_path: str = notebook
                notebook = Path(notebook).name  # e.g. 'notebook.ipynb'
            else:
                nb_input_file_path: str = self._find_template(notebook)
            nb_output_file_path = Path(output_path or '.').resolve() / f'{notebook.replace(".ipynb", "")}_output.ipynb'
            nb_output_file_path = str(nb_output_file_path)
            nb_output_file_paths.append(nb_output_file_path)
            
            print(f"Executing notebook: {notebook}")
            pm.execute_notebook(
                nb_input_file_path,
                nb_output_file_path,
                parameters=data
            )
            return
            
            if open_output:
                if editor == 'vscode':
                    if utils.is_command_available('code'):
                        subprocess.run(['code', nb_output_file_path])
                    else:
                        print("VSCode command 'code' is not available, cannot open the output notebook")
                elif editor == 'pycharm':
                    for cmd in ['charm', 'pycharm']:
                        if utils.is_command_available(cmd):
                            subprocess.run([cmd, nb_output_file_path])
                            break
                    else:
                        print("PyCharm command 'charm'/'pycharm' are both not available, cannot open the output notebook")
                else:
                    print(f"Editor '{editor}' is not supported, cannot open the output notebook")
            
            # TODO
            # if display:
            #     process = subprocess.Popen(['voila', f'--strip_sources={show_results_only}', nb_output_file_path])
            
            # TODO
            # if not save_output:
            #     print(f"Removing output notebook: {nb_output_file_path}")
            #     os.remove(nb_output_file_path)
                
        return nb_output_file_paths
        
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