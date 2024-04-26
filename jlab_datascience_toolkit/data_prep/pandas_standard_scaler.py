import pandas as pd
import os
import numpy as np
from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep
import inspect
import yaml

class PandasStandardScaler(JDSTDataPrep):
    """ Module performs standard scaling on Pandas DataFrames.
    """
    def __init__(self, config: dict = None, registry_config: dict = None):
        # Set default config
        self.config = dict(axis = 0)

	if registry_config is not None:
            self.config.update(registry_config)
        if config is not None:
            self.config.update(config)

        self.setup()

    @property
    def name(self):
        return "PandasStandardScaler_v0"

    def setup(self):
        self.mean = 0
        self.scale = 1
        
    def get_info(self):
        """ Prints this module's docstring. """
        print(inspect.getdoc(self)
    
    def save(self, path):
        pass

    def load(self, path):
        pass

    def save_config(self, path):
        pass

    def load_config(self, path):
        pass

    def run(self, data):
        pass

    def fit(self, data):
        pass

    def transform(self, data):
        pass

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        pass

    def run(self, data):
        pass

    def reverse(self, data):
        return self.inverse_tranform(data)
