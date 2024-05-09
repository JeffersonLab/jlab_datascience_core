from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep
from jlab_datascience_toolkit.utils.io import save_yaml_config, load_yaml_config
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import inspect
import yaml
import os

prep_log = logging.getLogger('Prep Logger')

def _fix_small_scales(scale, epsilon):
    """ Updates scale parameters below epsilon to 1 to prevent issues with small divisors

    Args:
        scale (_type_): Scale parameters to (potentially) fix
        epsilon (_type_): Smallest allowable value for scale parameters

    Returns:
        _type_: Updated scale parameters
    """
    return np.where(scale < epsilon, 1, scale)

class PandasStandardScaler(JDSTDataPrep):
    """ Module performs standard scaling on Pandas DataFrames.
    """
    def __init__(self, config: dict = None, registry_config: dict = None):
        # Set default config
        self.config = dict(
            axis = 0,
            epsilon = 1e-7,
            inplace = False
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
        self.mean = None
        self.var = None
        self.scale = None
        self.n_samples = 0
        
    def get_info(self):
        """ Prints this module's docstring. """
        print(inspect.getdoc(self))
    
    def save(self, path):
        os.makedirs(path)
        self.save_config(path)
        self.save_internal_state(path)

    def load(self, path):
        self.load_config(path)
        self.load_internal_state(path)

    def save_config(self, path, overwrite=False):
        """Save the module configuration to a folder at `path`

        Args:
            path (str): Location to save the module config yaml file
        """
        save_dir = Path(path)
        save_yaml_config(self.config, save_dir, overwrite)

    def load_config(self, path):
        """ Load the entire module state from `path`

        Args:
            path (str): Path to folder containing module files.
        """
        base_path = Path(path)
        self.config.update(load_yaml_config(base_path))
        self.setup()

    def save_internal_state(self, path):
        internal_state = dict(
            mean = self.mean,
            var = self.var,
            scale = self.scale,
            n_samples = self.n_samples
        )
        save_dir = Path(path)
        if not save_dir.exists():
            os.makedirs(save_dir)
        np.savez(save_dir.joinpath('scaler_state.npz'), **internal_state)

    def load_internal_state(self, path):
        save_dir = Path(path)
        internal_state = np.load(save_dir.joinpath('scaler_state.npz'))
        self.mean = internal_state['mean']
        self.var = internal_state['var']
        self.scale = internal_state['scale']
        self.n_samples = internal_state['n_samples']

    def run(self, data):
        if self.mean is None:
            prep_log.debug('Fitting new data on run()')
            self.fit(data)

        return self.transform(data)

    def reverse(self, data):
        return self.inverse_transform(data)

    def fit(self, data):
        # Since we do not modify data here, we can avoid a copy using np.asarray
        data_view = np.asarray(data)
        self.mean = np.mean(data_view, axis=self.config['axis'])
        self.var  = np.var(data_view, axis=self.config['axis'])
        self.scale = _fix_small_scales(np.sqrt(self.var), self.config['epsilon'])
        self.n_samples = data.shape[0]

    def transform(self, data):
        if self.mean is None:
            raise RuntimeError()
        data_view = np.array(data, copy=not self.config['inplace'])
        if self.config['axis'] is not None:
            data_rotated = np.rollaxis(data_view, self.config['axis'])
        else: data_rotated = data_view
        data_rotated -= self.mean
        data_rotated /= self.scale
        
        if self.config['inplace']:
            return

        output = data.copy()
        output.values[:] = data_view
        return output

    def inverse_transform(self, data):
        if self.mean is None:
            raise RuntimeError()
        data_view = np.array(data, copy=not self.config['inplace'])
        if self.config['axis'] is not None:
            data_rotated = np.rollaxis(data_view, self.config['axis'])
        else: data_rotated = data_view
        data_rotated *= self.scale
        data_rotated += self.mean
        
        if self.config['inplace']:
            return

        output = data.copy()
        output.values[:] = data_view
        return output

    def save_data(self, data):
        super().save_data()