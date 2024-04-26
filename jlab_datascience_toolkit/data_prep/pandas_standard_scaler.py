from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import numpy as np
import inspect
import yaml
import os

class PandasStandardScaler(JDSTDataPrep):
    """ Module performs standard scaling on Pandas DataFrames.
    """
    def __init__(self, config: dict = None, registry_config: dict = None):
        # Set default config
        self.config = dict(
            axis = 0,
            epsilon = 1e-6,
        )

    if registry_config is not None:
        self.config.update(registry_config)
    if config is not None:
        self.config.update(config)

    self.setup()

    @property
    def name(self):
        return "PandasStandardScaler_v0"

    def setup(self):
        self.scaler = StandardScaler().set_output(transform='pandas')
        
    def get_info(self):
        """ Prints this module's docstring. """
        print(inspect.getdoc(self))
    
    def save(self, path):
        self.save_config(path)
        self.save_internal_state(path)

    def load(self, path):
        self.load_config(path)
        self.load_internal_state(path)

    def save_config(self, path):
        """Save the module configuration to a folder at `path`

        Args:
            path (str): Location to save the module config yaml file
        """
        save_dir = Path(path)
        os.makedirs(save_dir)
        with open(save_dir.joinpath('config.yaml'), 'w') as f:
            yaml.safe_dump(self.config, f)

    def load_config(self, path):
        """ Load the entire module state from `path`

        Args:
            path (str): Path to folder containing module files.
        """
        base_path = Path(path)
        with open(base_path.joinpath('config.yaml'), 'r') as f:
            loaded_config = yaml.safe_load(f)

        self.config.update(loaded_config)
        self.setup()

    def save_internal_state(self, path):
        internal_state = self.scaler.__dict__
        # TODO save internal state to yaml

    def load_internal_state(self, path):
        # TODO load internal state from yaml
        for k, v in internal_state.items():
            setattr(self.scaler, k, v)

    def run(self, data):
        if self.mean is None:
            return self.fit_transform(data)

        return self.transform(data)

    def reverse(self, data):
        return self.inverse_tranform(data)

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)