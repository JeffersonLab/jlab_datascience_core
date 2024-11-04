from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
from jlab_datascience_toolkit.utils.io import save_yaml_config, load_yaml_config
from pathlib import Path
import pandas as pd
import logging
import yaml
import inspect
import os

parser_log = logging.getLogger('Parser Logger')

# Supported file formats
pandas_read_functions = dict(
    csv=pd.read_csv,
    feather=pd.read_feather,
    json=pd.read_json,
    pickle=pd.read_pickle
)

class Parser2DataFrame(JDSTDataParser):
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
            Arguments to be passed to the read function determined by `file_format`
        `concat_kwargs: dict = {}`
            Arguments to be passed to pd.concat()

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

    def __init__(self, config: dict = None, registry_config: dict = None):
        # It is important not to use default mutable arguments in python
        #   (lists/dictionaries), so we set config to None and update later

        # Priority for configurations is:
        # 1) config (intended for users)
        # 2) registry_config (intended only for the registry)
        # 3) defaults (set below)

        # Set default config
        self.config = dict(
            filepaths=[], 
            file_format='csv',
            read_kwargs = {},
            concat_kwargs = {},
        )

        # First update defaults with registry_configuration
        if registry_config is not None:
            parser_log.debug(f'Updating defaults with: {registry_config}')
            self.config.update(registry_config)

        # Now update configuration with new (user) configuration
        if config is not None:
            parser_log.debug(f'Updating registered config with: {config}')
            self.config.update(config)

        # To handle strings and lists of strings, we convert the former here
        if isinstance(self.config['filepaths'], str):
            self.config['filepaths'] = [self.config['filepaths']]

        self.setup()

    @property
    def name(self):
        return 'Parser2DataFrame_v0'

    def setup(self):
        # Set the correct reading function here
        self.read_function = pandas_read_functions.get(
            self.config['file_format'].lower(), None)

        if self.read_function is None:
            parser_log.error(
                    f'File format {self.config["file_format"]}'
                     'is not currently supported.')
            raise ValueError

    def get_info(self):
        """ Prints the docstring for the Parser2DataFrame module"""
        print(inspect.getdoc(self))

    def load(self, path: str):
        """ Load the entire module state from `path`

        Args:
            path (str): Path to folder containing module files.
        """
        base_path = Path(path)
        self.load_config(base_path)

    def save(self, path: str):
        """Save the entire module state to a folder at `path`

        Args:
            path (str): Location to save the module folder
        """
        save_dir = Path(path)
        os.makedirs(save_dir)
        self.save_config(save_dir)

    def load_data(self) -> pd.DataFrame:
        """ Loads all files listed in `config['filepaths']` 
        read_kwargs are passed to the appropriate pd.read_{file_format} function
        concat_kwargs are passed to pd.concat() after all files are read

        Returns:
            pd.DataFrame: A single DataFrame containing concatenated data
        """
        data_list = []
        for file in self.config['filepaths']:
            parser_log.debug(f'Loading {file} ...')
            data = self.read_function(
                file, 
                **self.config['read_kwargs'])
            data_list.append(data)

        # Check for empty data and return nothing if empty
        if not data_list:
            parser_log.warning(
                'load_data() returning None. This is probably not what you '
                'wanted. Ensure that your configuration includes the key '
                '"filepaths"')
            return 
        
        output = pd.concat(
            data_list, 
            **self.config['concat_kwargs'])
        
        return output

    def load_config(self, path: Path | str):
        self.config.update(load_yaml_config(path))
        self.setup()

    def save_config(self, path: Path | str, overwrite=False):
        """ Saves this modules configuration to the file specified by path
            If path is a directory, we save the configuration as config.yaml

        Args:
            path (Path | str): Location for saved configuration. Either a filename or directory is 
                acceptable.
        """
        save_yaml_config(self.config, path, overwrite)
    
    def save_data(self):
        return super().save_data()
