from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
from pathlib import Path
import pandas as pd
import logging
import yaml
import inspect
import os

csv_parser_log = logging.getLogger('CSV Reader to Pandas V0 Logger')


class CSV_Parser_to_Pandas(JDSTDataParser):
    """Reads a list of .csv files and concatenates them along a given axis.

    Optional intialization arguments: 
        `config: dict`
        `name: str`

    Required configuration keys: 
        `filepaths: str | list[str]`
    Optional configuration keys: 
        `axis: int = 0`
        `sep: str = ','`
        `read_args: dict = {}`
        `concat_args: dict = {'ignore_index: True}`


    Attributes
    ----------
    module_name : str
        Name of the module
    config: dict
        Configuration information

    Methods
    -------
    get_info()
        Prints this docstring
    load(path)
        Loads this module from `path/self.module_name`
    save(path)
        Saves this module from `path/self.module_name`
    load_data(path)
        Loads all files listed in `config['filepaths']` and concatenates them 
        along the `config['axis']` axis
    save_data(path)
        Does nothing
    load_config(path)
        Calls load(path)
    save_config(path)
        Calls save(path)

    """

    def __init__(self, config: dict = None, name: str = "CSV Reader to Pandas V0"):
        # It is important not to use default mutable arguments in python
        #   (lists/dictionaries), so we set config to None and update later
        self.module_name = name

        # Set default config
        self.config = dict(
            filepaths=[], 
            axis=0,
            sep=',',
            read_args = {},
            concat_args = {'ignore_index': True},
        )
        # Update configuration with new configuration
        if config is not None:
            self.config.update(config)

        # To handle strings and lists of strings, we convert the former here
        if isinstance(self.config['filepaths'], str):
            self.config['filepaths'] = [self.config['filepaths']]

    def get_info(self):
        """ Prints the docstring for the AIDAPTCSVReaderV0 module"""
        print(inspect.getdoc(self))

    def load(self, path: str):
        """ Load the entire module state from `path`

        Args:
            path (str): Path to folder containing module files.
        """
        base_path = Path(path)
        save_dir = base_path.joinpath(self.module_name)
        with open(save_dir.joinpath('config.yaml'), 'r') as f:
            loaded_config = yaml.safe_load(f)

        self.config.update(loaded_config)

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
        """ Loads all files listed in `config['filepaths']` and concatenates 
        them along the `config['axis']` axis

        Returns:
            pd.DataFrame: A single DataFrame containing concatenated data
        """
        data_list = []
        for file in self.config['filepaths']:
            csv_parser_log.debug(f'Loading {file} ...')
            data = pd.read_csv(
                file, 
                sep=self.config['sep'], 
                **self.config['read_args'])
            data_list.append(data)

        # Check for empty data and return nothing if empty
        if not data_list:
            csv_parser_log.warn(
                'load_data() returning None. This is probably not what you '
                'wanted. Ensure that your configuration includes the key '
                '"filepaths"')
            return 
        
        output = pd.concat(data_list, axis=self.config['axis'], **self.config['concat_args'])
        return output

    # Unimplemented functions
    def save_data(self, path: str):
        csv_parser_log.warning(
            'save_data() is currently unimplemented.')
        pass

    def load_config(self, path: str):
        csv_parser_log.debug(
            'load_config() is currently unimplemented. '
            'Calling load()...')
        return self.load(path)

    def save_config(self, path: str):
        csv_parser_log.debug(
            'save_config() is currently unimplemented.'
            ' Calling save()...?')
        return self.save(path)