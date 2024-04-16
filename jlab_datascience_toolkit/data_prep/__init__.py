from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

# Min Max Scaler:
register(
    id="NumpyMinMaxScaler_v0",
    entry_point="jlab_datascience_toolkit.data_prep.numpy_minmax_scaler:NumpyMinMaxScaler"
)

from jlab_datascience_toolkit.data_prep.numpy_minmax_scaler import NumpyMinMaxScaler

# Numpy Linear Scaler:
register(
    id="NumpyLinearScaler_v0",
    entry_point="jlab_datascience_toolkit.data_prep.numpy_linear_scaler:NumpyLinearScaler"
)

from jlab_datascience_toolkit.data_prep.numpy_linear_scaler import NumpyLinearScaler