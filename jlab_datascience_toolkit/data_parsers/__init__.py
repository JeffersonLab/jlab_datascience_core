from jlab_datascience_toolkit.utils.registration import (
    register,
    make,
    list_registered_modules,
)

register(
    id="NumpyParser_v0",
    entry_point="jlab_datascience_toolkit.data_parsers.numpy_parser:NumpyParser",
)

from jlab_datascience_toolkit.data_parsers.numpy_parser import NumpyParser

register(
    id="CSVParser_v0",
    entry_point="jlab_datascience_toolkit.data_parsers.parser_to_dataframe:Parser2DataFrame",
    kwargs={"registry_config": {"file_format": "csv"}},
)

register(
    id="FamousDatasets_v0",
    entry_point="jlab_datascience_toolkit.data_parsers.famous_datasets_v0:FamousDatasetsV0",
)