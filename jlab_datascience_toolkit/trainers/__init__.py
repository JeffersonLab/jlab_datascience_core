from jlab_datascience_toolkit.utils.registration import (
    register,
    make,
    list_registered_modules,
)

register(
    id="KerasTrainer_v0",
    entry_point="jlab_datascience_toolkit.trainers.keras_trainer_v0:Trainer",
)
