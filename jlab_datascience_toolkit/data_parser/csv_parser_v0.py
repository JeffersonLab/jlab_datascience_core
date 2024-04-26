from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
import jlab_datascience_toolkit.utils.parser_utilities as utils
from pathlib import Path
import pandas as pd
import logging
import yaml
import inspect
import os

parser_log = logging.getLogger('CSVParser Logger')

class CSVParser(JDSTDataParser):
    """Reads a list of CSV files and concatenates them in a Pandas DataFrame.

    Intialization arguments: 
        `config: dict`

    Optional configuration keys: 
        `filepaths: str | list[str]`
            Paths to files the module should parse. Defaults to `[]` which 
            produces a warning when load_data() is called.
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

        # Set default config
        self.config = dict(
            filepaths=[], 
            read_kwargs = {},
            concat_kwargs = {},
        )
        # Update configuration with new configuration
        if config is not None:
            self.config.update(config)

        # To handle strings and lists of strings, we convert the former here
        if isinstance(self.config['filepaths'], str):
            self.config['filepaths'] = [self.config['filepaths']]

    @property
    def name(self):
        return 'CSVParser_v0'

    def get_info(self):
        """ Prints the docstring for the CSVParser module"""
        print(inspect.getdoc(self))

    def load(self, path: str):
        """ Load the entire module state from `path`

        Args:
            path (str): Path to folder containing module files.
        """
        self.config.update(utils.load_yaml_config(path))

    def save(self, path: str):
        """Save the entire module state to a folder at `path`

        Args:
            path (str): Location to save the module folder
        """
        utils.save_config_to_yaml(self.config, path)

    def load_data(self) -> pd.DataFrame:
        """ Loads all files listed in `config['filepaths']` 
        read_kwargs are passed to the appropriate pd.read_{file_format} function
        concat_kwargs are passed to pd.concat() after all files are read

        Returns:
            pd.DataFrame: A single DataFrame containing concatenated data
        """
        data_list = utils.read_data_to_pandas(
            filepaths=self.config['filepaths'],
            file_format='csv',
            **self.config['read_kwargs']
        )


        # Check for empty data and return nothing if empty
        if not data_list:
            parser_log.warning(
                'load_data() returning None. This is probably not what you '
                'wanted. Ensure that your configuration includes the key '
                '"filepaths"')
            return 
        
        return pd.concat(data_list, **self.config['concat_kwargs'])
        
    def load_config(self, path: str):
        parser_log.debug('Calling load()...')
        return self.load(path)

    def save_config(self, path: str):
        parser_log.debug('Calling save()...')
        return self.save(path)
    
    def save_data(self):
        return super().save_data()