from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

register('base_tensorflow_model_v1', entry_point='jlab_datascience_toolkit.models.tensorflow_model:TensorflowModel')