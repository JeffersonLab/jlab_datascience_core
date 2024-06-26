from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

register(
    id="NumpyParser_v0",
    entry_point="jlab_datascience_toolkit.data_parser.numpy_parser:NumpyParser"
)

from jlab_datascience_toolkit.data_parser.numpy_parser import NumpyParser

register(
    id='CSVParser_v0',
    entry_point="jlab_datascience_toolkit.data_parser.parser_to_dataframe:Parser2DataFrame",
    kwargs={'registry_config': {'file_format': 'csv'}}
)