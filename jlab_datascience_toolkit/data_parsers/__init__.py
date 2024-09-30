from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

# Numpy Parser:
register(
    id="NumpyParser_v0",
    entry_point="jlab_datascience_toolkit.data_parsers.numpy_parser:NumpyParser"
)

# Pandas Parser:
register(
    id="PandasParser_v0",
    entry_point="jlab_datascience_toolkit.data_parsers.pandas_parser_v0:PandasParser"
)

# Image to Numpy parser:
register(
    id="ImageToNumpyParser_v0",
    entry_point="jlab_datascience_toolkit.data_parsers.image_to_numpy_parser:ImageToNumpyParser"
)

# MNIST Data parser:
register(
    id="MNISTDataParser_v0",
    entry_point="jlab_datascience_toolkit.data_parsers.mnist_data_parser:MNISTDataParser"
)
