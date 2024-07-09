from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
from pathlib import Path
import pandas as pd
import logging
import yaml
import inspect
import os

pandas_parser_log = logging.getLogger('PandasParser_v0 Logger')

# Supported file formats
pandas_read_functions = dict(
    csv=pd.read_csv,
    feather=pd.read_feather,
    json=pd.read_json,
    pickle=pd.read_pickle
)

class PandasParser(JDSTDataParser):
    """Reads a list of files and concatenates them in a Pandas DataFrame.

    Intialization arguments: 
        `config: dict`

    Optional configuration keys: 
        `filepaths: str | list[str]`
            Paths to files the module should parse. Defaults to `[]` which 
            produces a warning when load_data() is called.
        `file_format: str = 'csv',
            Format of files to parse. Currently supports csv, feather, json
            and pickle. Defaults to csv
        `read_kwargs: dict = {}`
            Arguments to be passed 
        `concat_kwargs: dict = {}`

    Attributes
    ----------
    name : str
        Name of the module
    config: dict
        Configuration information

    Methods
    -------
    get_info()
        Prints this docstring
    load(path)
        Loads this module from `path`
    save(path)
        Saves this module to `path`
    load_data(path)
        Loads all files listed in `config['filepaths']` and concatenates them
    save_data(path)
        Does nothing
    load_config(path)
        Calls load(path)
    save_config(path)
        Calls save(path)

    """

    def __init__(self, config: dict = None):
        # It is important not to use default mutable arguments in python
        #   (lists/dictionaries), so we set config to None and update later

        self.module_name = "pandas_parser"

        # Set default config
        self.config = dict(
            filepaths=[], 
            file_format='csv',
            read_kwargs = {},
            concat_kwargs = {},
        )
        # Update configuration with new configuration
        if config is not None:
            self.config.update(config)

        # To handle strings and lists of strings, we convert the former here
        if isinstance(self.config['filepaths'], str):
            self.config['filepaths'] = [self.config['filepaths']]

        self.setup()

    @property
    def name(self):
        return 'PandasParser_v0'

    def setup(self):
        # Set the correct reading function here
        self.read_function = pandas_read_functions.get(
            self.config['file_format'].lower(), None)

        if self.read_function is None:
            pandas_parser_log.error(
                    f'File format {self.config["file_format"]}'
                     'is not currently supported.')
            raise ValueError
        
    def check_input_data_type(self,input_data):
        if isinstance(input_data,list) == False:
            logging.error(f">>> {self.name}: The input data type {type(input_data)} is not a list. Please correct. Going to returne None <<<")
            return False
        else:
            if len(input_data) > 0:
                return True
            else:
                logging.error(f">>> {self.name}: The list of filepaths your provided {input_data} seems to be empty. Please check your configuration. Going to return None <<<")
                return False

    def get_info(self):
        """ Prints the docstring for the PandasParser module"""
        print(inspect.getdoc(self))

    def load(self, path: str):
        """ Load the entire module state from `path`

        Args:
            path (str): Path to folder containing module files.
        """
        base_path = Path(path)
        with open(base_path.joinpath('config.yaml'), 'r') as f:
            loaded_config = yaml.safe_load(f)

        self.config.update(loaded_config)
        self.setup()

    def save(self, path: str):
        """Save the entire module state to a folder at `path`

        Args:
            path (str): Location to save the module folder
        """
        save_dir = Path(path)
        os.makedirs(save_dir)
        with open(save_dir.joinpath('config.yaml'), 'w') as f:
            yaml.safe_dump(self.config, f)

    def load_data(self) -> pd.DataFrame:
        """ Loads all files listed in `config['filepaths']` 
        read_kwargs are passed to the appropriate pd.read_{file_format} function
        concat_kwargs are passed to pd.concat() after all files are read

        Returns:
            pd.DataFrame: A single DataFrame containing concatenated data
        """

        if self.check_input_data_type(self.config['filepaths']) == True:

            data_list = []
            for file in self.config['filepaths']:
               pandas_parser_log.debug(f'Loading {file} ...')
               data = self.read_function(
                   file, 
                   **self.config['read_kwargs'])
               data_list.append(data)

            return pd.concat(
            data_list, 
            **self.config['concat_kwargs'])
        
        
        return None

    def load_config(self, path: str):
        pandas_parser_log.debug('Calling load()...')
        return self.load(path)

    def save_config(self, path: str):
        pandas_parser_log.debug('Calling save()...')
        return self.save(path)
    
    def save_data(self):
        return super().save_data()