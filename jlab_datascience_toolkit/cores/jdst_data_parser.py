from jlab_datascience_toolkit.cores.jdst_module import JDSTModule
from abc import ABC, abstractmethod


class JDSTDataParser(JDSTModule, ABC):
    """
    Base class for data parsing. This class inherits from the module base class.
    """

    # Load and save the data:
    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def save_data(self):
        raise NotImplementedError
