import os
import numpy as np
from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep
from aidapt_toolkit.utils.config_utils import verify_config
import inspect
import yaml


class PandasMinMaxScaler(JDSTDataPrep):
    """Module performs a min/max scaling on Pandas DataFrames.

    Assumes data is formatted as (samples, features, ...) and that scaling is 
    desired over all dimensions (1 scaler per feature dimension)
    """

    def __init__(self, config, name='numpy_minmax_scaler') -> None:
        self.module_name = name
        self.required_config_keys = ['feature_range']
        verify_config(config, self.required_config_keys)
        self.config = config

        self.feature_range = self.config['feature_range']

        if self.feature_range[1] <= self.feature_range[0]:
            raise ValueError(
                'Feature range must contain two entries in increasing order.')

    def get_info(self):
        print(inspect.getdoc(self))

    def load_config(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        verify_config(config, self.required_config_keys)
        self.config = config

    def save_config(self, path):
        with open(path, 'w') as f:
            yaml.safe_dump(self.config, f)

    def train(self, data: np.ndarray):
        x = data
        if data.ndim == 1:
            x = x[:, np.newaxis]

        self.data_min = np.array(np.min(x, axis=0))
        self.data_max = np.array(np.max(x, axis=0))
        self.data_range = self.data_max - self.data_min
        self.data_range = np.where(
            self.data_range < 1e-5,
            np.ones_like(self.data_range),
            self.data_range
        )

    def run(self, data: np.ndarray):
        x = data
        if data.ndim == 1:
            x = x[:, np.newaxis]

        data_zero_one = (x - self.data_min) / self.data_range
        data_scaled = data_zero_one * \
            (self.feature_range[1] - self.feature_range[0])
        data_scaled = data_scaled + self.feature_range[0]
        return data_scaled

    def reverse(self, data: np.ndarray):
        x = data
        if data.ndim == 1:
            x = x[:, np.newaxis]

        data_zero_one = (x - self.feature_range[0]) / \
            (self.feature_range[1] - self.feature_range[0])
        data_unscaled = data_zero_one * self.data_range + self.data_min
        return data_unscaled

    def save_data(self, data: np.ndarray):
        pass

    def save(self, path):
        os.makedirs(path)
        fullpath = os.path.join(path, f'minmax_scaler_params.npz')
        np.savez(fullpath, data_min=self.data_min,
                 data_max=self.data_max, data_range=self.data_range)

    def load(self, path):
        fullpath = os.path.join(path, f'minmax_scaler_params.npz')
        arg_dict = np.load(fullpath)
        self.data_min = arg_dict['data_min']
        self.data_max = arg_dict['data_max']
        self.data_range = arg_dict['data_range']
