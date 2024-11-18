from jlab_datascience_toolkit.utils.registration import (
    register,
    make,
    list_registered_modules,
)

register(
    id="NumpyMinMaxScaler_v0",
    entry_point="jlab_datascience_toolkit.data_prep.numpy_minmax_scaler:NumpyMinMaxScaler",
)

from jlab_datascience_toolkit.data_prep.numpy_minmax_scaler import NumpyMinMaxScaler

register(
    id="PandasStandardScaler_v0",
    entry_point="jlab_datascience_toolkit.data_prep.pandas_standard_scaler:PandasStandardScaler",
)

register(
    id="SplitDataFrame_v0",
    entry_point="jlab_datascience_toolkit.data_prep.split_dataframe_v0:SplitDataFrame",
)
