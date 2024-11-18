import yaml
from pathlib import Path
import tempfile
import logging
import sys
from typing import Union

io_log = logging.getLogger('io_log')

def save_yaml_config(config: dict, path: Union[str, Path], overwrite: bool = False):
    """ Saves configuration dictionary to a yaml file

    Args:
        config (dict): Dictionary to save
        path (str | Path): Location to save configuration. 
            If `path` does not exist, it will be created.
            If `path` is a directory, the configuration will be saved to config.yaml
            If `path` is a filename, the configuration will be saved to that filename
        overwrite (bool, optional): If True, the passed configuration will overwrite any existing
            file with the same `path`. Defaults to False.

    Raises:
        FileExistsError: If `path` exists and `overwrite==False` a FileExistsError will be raised.
    """
    path = Path(path)

    if path.is_dir():
        io_log.info('path.is_dir() == True')
        path = path.joinpath('config.yaml')

    path.parent.mkdir(exist_ok=True)

    if path.exists() and not overwrite:
        io_log.error(f'File {path} exists without overwrite flag set')
        raise FileExistsError('File already exists. Set overwrite=True if you would like to overwrite it.')
    
    with open(path, 'w') as f:
        io_log.info(f'Writing config to {path}')
        yaml.safe_dump(config, f)

def load_yaml_config(path: Union[str, Path]):
    path = Path(path)
    if path.is_dir():
        path = path.joinpath('config.yaml')
        
    if not path.exists():
        io_log.error(f'Configuration file {path} not found.')
        raise FileNotFoundError(f'Configuration file {path} not found.')

    with open(path, 'r') as f:
        return yaml.safe_load(f)
