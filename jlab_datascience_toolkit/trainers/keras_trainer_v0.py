import yaml
import logging
import keras
import inspect
from jlab_datascience_toolkit.core.jdst_trainer import JDSTTrainer


trainer_log = logging.getLogger("Trainer Logger")


class Trainer(JDSTTrainer):
    """
    Trains a given JDSTModel object "model" that has a keras.Model attribute (i.e., model.model is a keras.Model) on given data and training configurations.

    Arguments of keras.Model.fit are divided into:
    1. Configurations passed in "configs" of Trainer.__init__ method. These are:
        a) batch_size=None
        b) epochs=1
        c) verbose="auto"
        d) callbacks=[]
        e) validation_split=0.0
        f) shuffle=True
        g) class_weight=None ===> WARNING: If passed, make sure class indices match those in targets "y"
        h) initial_epoch=0
        i) steps_per_epoch=None
        j) validation_steps=None
        k) validation_batch_size=None
        l) validation_freq=1
    2. Data (not configurations) passed to Trainer.fit method. These are:
        a) x
        b) y
        c) validation_data
        d) sample_weight

    In addition to the list in (1.), two additional items are part of the training configurations:
    1) "loss_configs"
    2) "optimizer_configs"

    Methods
    -------
    init()
        Sets the configurations of training
    fit()
        Fits given model on given training data
    get_info()
        Prints this docstring
    save_config()
        Saves object's configurations to a given path
    load_config()
        Satatic method loading configurations from a given path
    """

    def __init__(self, configs: dict):
        self.configs = configs
        self.settings = (
            configs.copy()
        )  # Must be separate from configs as it can include actual keras callback objects
        self.settings.pop("registered_name")

        # 1) Loss
        loss_configs = self.settings.pop("loss_configs")
        loss_type = loss_configs.pop("loss_type")
        if loss_type == "CategoricalCrossentropy":
            self.loss = keras.losses.CategoricalCrossentropy(**loss_configs)
        elif loss_type == "SparseCategoricalCrossentropy":
            self.loss = keras.losses.SparseCategoricalCrossentropy(**loss_configs)
        else:
            raise NameError(f"Unrecognized loss_type ({loss_type}) !!!")

        # 2) Optimizer
        optimizer_configs = self.settings.pop("optimizer_configs")
        optimizer_type = optimizer_configs.pop("optimizer_type")
        if optimizer_type == "Adam":
            self.optimizer = keras.optimizers.Adam(**optimizer_configs)
        elif optimizer_type == "RMSprop":
            self.optimizer = keras.optimizers.RMSprop(**optimizer_configs)
        else:
            raise NameError(f"Unrecognized optimizer_type ({optimizer_type}) !!!")

        # 3) OPTIONAL Callbacks
        callbacks = []
        for callback_configs in self.settings.get("callbacks", []):
            callback_configs = callback_configs.copy()
            callback_type = callback_configs.pop("callback_type")
            if callback_type == "EarlyStopping":
                callbacks.append(keras.callbacks.EarlyStopping(**callback_configs))
            elif callback_type == "ReduceLROnPlateau":
                callbacks.append(keras.callbacks.ReduceLROnPlateau(**callback_configs))
            else:
                raise NameError("Unrecognized callback_type !!!")
        self.settings["callbacks"] = None if len(callbacks) == 0 else callbacks

        # 4) Check on "class_weight"
        if self.settings.get("class_weight", None) is not None:
            trainer_log.warning(
                'Make sure indices of classes in "class_weight" match indices of "y" !'
            )

    def fit(self, model, x=None, y=None, validation_data=None, sample_weight=None):
        model.model.compile(optimizer=self.optimizer, loss=self.loss)
        history = model.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            sample_weight=sample_weight,
            **self.settings,
        )
        return history

    def get_info(self):
        """Prints this module's docstring."""
        print(inspect.getdoc(self))

    def save_config(self, path: str):
        assert path.endswith(".yaml")
        with open(path, "w") as file:
            yaml.safe_dump(self.configs, file)

    @staticmethod
    def load_config(path: str):
        assert path.endswith(".yaml")
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
