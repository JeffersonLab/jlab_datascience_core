from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

# Keras CNN Autoencoder:
register(
    id="KerasCNNAE_v0",
    entry_point="jlab_datascience_toolkit.model.keras_cnn_ae:KerasCNNAE"
)

from jlab_datascience_toolkit.model.keras_cnn_ae import KerasCNNAE
