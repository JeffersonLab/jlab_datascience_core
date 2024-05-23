import inspect
from pathlib import Path

import keras
import matplotlib.pyplot as plt

from jlab_datascience_toolkit.core.jdst_model import JDSTModel
from jlab_datascience_toolkit.utils.io import load_yaml_config, save_yaml_config


class TensorflowModel(JDSTModel):
    """A basic Tensorflow Model.

    <long form description here...>
    """

    def __init__(self, config: dict = None, registry_config: dict = None):
        self.config = {
            "model_config": {
                "module": "keras",
                "class_name": "Sequential",
                "config": {
                    "name": "sequential_1",
                    "trainable": True,
                    "dtype": "float32",
                    "layers": [],
                },
                "registered_name": None,
            },
            "compile_config": {
                "optimizer": "rmsprop",
                "loss": None,
                "loss_weights": None,
                "metrics": None,
                "weighted_metrics": None,
                "run_eagerly": False,
                "steps_per_execution": 1,
                "jit_compile": False,
            },
        }

        if registry_config is not None:
            self.config.update(registry_config)
        if config is not None:
            self.config.update(config)

        self.setup()

    @property
    def name(self):
        return "TensorflowMLPModel_v0"

    def get_info(self):
        """Prints the docstring for the Tensorflow Model module."""
        print(inspect.getdoc(self))

    def load_config(self, path: Path | str):
        """Loads a configuration for this module from the file specified by path.

            The loaded configuration updates the currently loaded config, not the default.
            If path is a directory, the configuration is assumed to be saved as config.yaml.

        Args:
            path (Path | str): Location for saved configuration. Either a filename or directory is
                acceptable.
        """
        self.config.update(load_yaml_config(path))
        self.setup()

    def save_config(self, path: Path | str, overwrite=False):
        """Saves this modules configuration to the file specified by path.
            If path is a directory, we save the configuration as config.yaml

        Args:
            path (Path | str): Location for saved configuration. Either a filename or directory is
                acceptable.
        """
        save_yaml_config(self.config, path, overwrite)

    def save(self, path: Path | str):
        path = Path(path)
        self.save_config(path)
        self.model.save(path.joinpath("model.keras"))

    def load(self, path: Path | str):
        path = Path(path)
        self.load_config(path)
        keras.models.load_model(path.joinpath("model.keras"))

    def setup(self):
        self.model = keras.saving.deserialize_keras_object(self.config["model_config"])
        if not self.model.compiled:
            self.model.compile_from_config(self.config["compile_config"])

    def train(self, x, y, **kwargs):
        self.history = self.model.fit(
            x,
            y,
            **kwargs,
        )

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def analysis(self):
        hist = self.history.history
        for key in hist:
            plt.plot(hist[key], label=key)
        plt.legend()
        plt.show()
