from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

register(
    id="KerasMLP_v0",
    entry_point="jlab_datascience_toolkit.models.keras_mlp_v0:KerasMLP"
)
