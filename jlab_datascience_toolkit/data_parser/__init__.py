from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

# Numpy Parser:
register(
    id="NumpyParser_v0",
    entry_point="jlab_datascience_toolkit.data_parser.numpy_parser:NumpyParser"
)

from jlab_datascience_toolkit.data_parser.numpy_parser import NumpyParser

# Pandas Parser:
register(
    id="PandasParser_v0",
    entry_point="jlab_datascience_toolkit.data_parser.pandas_parser_v0:PandasParser"
)

# Image to Numpy parser:
register(
    id="ImageToNumpyParser_v0",
    entry_point="jlab_datascience_toolkit.data_parser.image_to_numpy_parser:ImageToNumpyParser"
)

from jlab_datascience_toolkit.data_parser.image_to_numpy_parser import ImageToNumpyParser

# MNIST Data parser:
register(
    id="MNISTDataParser_v0",
    entry_point="jlab_datascience_toolkit.data_parser.mnist_data_parser:MNISTDataParser"
)

from jlab_datascience_toolkit.data_parser.mnist_data_parser import MNISTDataParser