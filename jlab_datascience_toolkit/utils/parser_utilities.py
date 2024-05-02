import os
import pathlib
import yaml
import pandas as pd

def save_config_to_yaml(config, path):
    save_path = pathlib.Path(path)
    os.makedirs(save_path)
    with open(save_path.joinpath('config.yaml'), 'w') as f:
        yaml.safe_dump(self.config, f)

def load_yaml_config(path):
    base_path = Path(path)
    with open(base_path.joinpath('config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_data_to_pandas(filepaths: list, file_format: str, **kwargs) -> pd.DataFrame:
        """ Loads all files listed in filepaths and reads them.
        All kwargs other than filepaths and file_format will be passed to the read_function
        for its associated file_format

        Returns:
            pd.DataFrame: A single DataFrame containing list of dataframes
        """

        # Supported file formats
        read_functions = dict(
            csv=pd.read_csv,
            feather=pd.read_feather,
            json=pd.read_json,
            pickle=pd.read_pickle
        )

        data_list = []
        read_function = read_functions[file_format]
        for file in filepaths:
            data = read_function(file, **kwargs)
            data_list.append(data)

        return data_list