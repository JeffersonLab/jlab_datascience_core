import yaml
from pathlib import Path
import tempfile
import logging
import sys

io_log = logging.getLogger('io_log')

def save_config(config: dict, path: str | Path, overwrite: bool = False):
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

def load_config(path: str | Path):
    path = Path(path)
    if path.is_dir():
        path = path.joinpath('config.yaml')
        
    if not path.exists():
        io_log.error(f'Configuration file {path} not found.')
        raise FileNotFoundError(f'Configuration file {path} not found.')

    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    config = {'name': 'test', 'scale': 1, 'list_example': [0.1, 1.2, 2.3]}

    test_path = tempfile.mkdtemp()
    io_log.info(f'Made directory at {test_path}')

    # Testing with fake directory
    save_config(config, test_path, False)

    # File should exist now, so we should get an error
    try: 
        save_config(config, test_path, False)
    except:
        io_log.info('save_config() raised error')

    test_file = tempfile.mkstemp(suffix='.yaml')[1]
    io_log.info(f'Made file at {test_file}')

    # File should exist now, so we should get an error
    try: 
        save_config(config, test_file, False)
    except:
        io_log.info('save_config() raised error')
    
    # With overwrite flag, we should write the config to the file
    save_config(config, test_file, True)

    # Log saved configurations
    io_log.info(f'Saved config == {load_config(test_file)}')
    io_log.info(f'Saved config == {load_config(Path(test_path).joinpath("config.yaml"))}')