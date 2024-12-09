from jlab_datascience_toolkit.cores.jdst_data_prep import JDSTDataPrep
from jlab_datascience_toolkit.utils.io import save_yaml_config, load_yaml_config
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import inspect
import yaml
import os

prep_log = logging.getLogger("Prep Logger")


def _fix_small_scales(scale, epsilon):
    """Updates scale parameters below epsilon to 1 to prevent issues with small divisors

    Args:
        scale (array_like): Scale parameters to (potentially) fix
        epsilon (float): Smallest allowable value for scale parameters

    Returns:
        array_like: Updated scale parameters
    """
    return np.where(scale < epsilon, 1, scale)


class PandasStandardScaler(JDSTDataPrep):
    """Module performs standard scaling on Pandas DataFrames.

    Intialization arguments:
        config: dict

    Optional configuration keys:
        axis: int = 0
            Axis to perform scaling on. Accepts 0,1 or None. Defaults to 0.
        epsilon: float = 1e-7
            Smallest allowable value for the standard deviation. Defaults to 1e-7.
            If smaller than epsilon, the output variance will not be modified.
            This avoids exploding small noise variance values.
        inplace: bool = False
            If True, operations modify the original DataFrame. Defaults to False.

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
        Loads this module (including fit scaler parameters) from `path`
    save(path)
        Saves this module (including fit scaler parameters) to `path`
    load_config(path)
        Loads a configuration file. Scaler parameters will be fit to new data.
    save_config(path)
        Calls `save(path)`
    run(data)
        Performs standard scaling on `data`. If the scaler has not been previously
        fit, the scaler parameters will be fit to `data`. Otherwise, the scaling
        will utilize mean and variance information from the most recent `fit()` call.
    fit(data)
        Sets scaler parameters for mean and variance based on `data`
    reverse(data)
        Performs inverse scaling on `data`.
    save_data(path)
        Does nothing.

    """

    def __init__(self, config: dict = None, registry_config: dict = None):
        # Set default config
        self.config = dict(axis=0, epsilon=1e-7, inplace=False)

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
        """Prints this module's docstring."""
        print(inspect.getdoc(self))

    def save(self, path: str):
        """Save entire module to a folder at `path`

        Args:
            path (str): Location to save the module. This path must not currently exist.
        """
        os.makedirs(path)
        self.save_config(path)
        self.save_internal_state(path)

    def load(self, path: str):
        """Load entire saved module from `path`

        Args:
            path (str): Directory to load module from. Should include a config.yaml
                and scaler_state.npz files.
        """
        self.load_config(path)
        self.load_internal_state(path)

    def save_config(self, path: str, overwrite: bool = False):
        """Save the module configuration to a folder at `path`

        Args:
            path (str): Location to save the module config yaml file
            overwrite (bool, optional): If True, overwrites file at path if it exists.
                Defaults to False.
        """
        save_dir = Path(path)
        save_yaml_config(self.config, save_dir, overwrite)

    def load_config(self, path: str):
        """Load the entire module state from `path`

        Args:
            path (str): Path to folder containing module files.
        """
        base_path = Path(path)
        self.config.update(load_yaml_config(base_path))
        self.setup()

    def save_internal_state(self, path: str):
        internal_state = dict(
            mean=self.mean, var=self.var, scale=self.scale, n_samples=self.n_samples
        )
        save_dir = Path(path)
        if not save_dir.exists():
            os.makedirs(save_dir)
        np.savez(save_dir.joinpath("scaler_state.npz"), **internal_state)

    def load_internal_state(self, path: str):
        save_dir = Path(path)
        internal_state = np.load(save_dir.joinpath("scaler_state.npz"))
        self.mean = internal_state["mean"]
        self.var = internal_state["var"]
        self.scale = internal_state["scale"]
        self.n_samples = internal_state["n_samples"]

    def run(self, data: pd.DataFrame):
        if self.mean is None:
            prep_log.debug("Fitting new data on run()")
            self.fit(data)

        return self.transform(data)

    def reverse(self, data: pd.DataFrame):
        """Performs inverse scaling on `data`

        Args:
            data (pd.DataFrame): Data to perform inverse scaling on.

        Returns:
            pd.DataFrame: Inverse scaled DataFrame
        """
        return self.inverse_transform(data)

    def fit(self, data: pd.DataFrame):
        """Sets internal scaler parameters based on the mean and variance of `data`

        Args:
            data (pd.DataFrame): DataFrame used to fit the scaler
        """
        # Since we do not modify data here, we can avoid a copy using np.asarray
        data_view = np.asarray(data)
        self.mean = np.mean(data_view, axis=self.config["axis"])
        self.var = np.var(data_view, axis=self.config["axis"])
        self.scale = _fix_small_scales(np.sqrt(self.var), self.config["epsilon"])
        self.n_samples = data.shape[0]

    def transform(self, data):
        if self.mean is None:
            raise RuntimeError()
        data_view = np.array(data, copy=not self.config["inplace"])
        if self.config["axis"] is not None:
            data_rotated = np.rollaxis(data_view, self.config["axis"])
        else:
            data_rotated = data_view
        data_rotated -= self.mean
        data_rotated /= self.scale

        if self.config["inplace"]:
            return

        output = data.copy()
        output.values[:] = data_view
        return output

    def inverse_transform(self, data):
        if self.mean is None:
            raise RuntimeError()
        data_view = np.array(data, copy=not self.config["inplace"])
        if self.config["axis"] is not None:
            data_rotated = np.rollaxis(data_view, self.config["axis"])
        else:
            data_rotated = data_view
        data_rotated *= self.scale
        data_rotated += self.mean

        if self.config["inplace"]:
            return

        output = data.copy()
        output.values[:] = data_view
        return output

    def save_data(self, data):
        super().save_data()
