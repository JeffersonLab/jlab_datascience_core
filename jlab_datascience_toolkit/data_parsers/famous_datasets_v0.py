from jlab_datascience_toolkit.cores.jdst_data_parser import JDSTDataParser
import seaborn as sns
import inspect
import yaml


class FamousDatasetsV0(JDSTDataParser):
    """Returns one of the example famous datasets such as iris.

    Intialization arguments:
        `config: dict`
    
    Mandatory configuration key:
        `dataset_name: str`
            Name of an example dataset such as "iris"

    Optional configuration keys:
        Other configurations expected by sns.load_dataset for example

    Attributes
    ----------
    name : str
        Name of the module
    config: dict
        Configuration information

    Methods
    -------
    init()
        Sets the configurations of parser
    load_data()
        Returns data based on the configurations
    get_info()
        Prints this docstring
    save_config(path)
        Calls save(path)
    load_config(path)
        Calls load(path)
    load()
        Does nothing
    save()
        Does nothing
    save_data()
        Does nothing
    """
    def __init__(self, configs: dict):
        self.configs = configs
        self.dataset_name = configs['dataset_name']
        self.settings = {k: v for k, v in configs.items() if k not in {'dataset_name', 'registered_name'}}
    
    def load_data(self):
        if self.dataset_name == 'iris':
            return sns.load_dataset('iris', **self.settings)
        else:
            raise NameError(f'Dataset "{self.dataset_name}" must be one of ["iris"] !!!')
    
    def get_info(self):
        """Prints the docstring for the Parser2DataFrame module"""
        print(inspect.getdoc(self))
    
    def save_config(self, path: str):
        assert path.endswith(".yaml")
        with open(path, "w") as file:
            yaml.safe_dump(self.configs, file)

    @staticmethod
    def load_config(path: str):
        assert path.endswith(".yaml")
        with open(path, "r") as file:
            return yaml.safe_load(file)
    
    def load(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError

    def save_data(self):
        raise NotImplementedError