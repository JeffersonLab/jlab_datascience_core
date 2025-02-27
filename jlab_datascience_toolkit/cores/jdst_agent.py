from jlab_datascience_toolkit.cores.jdst_module import JDSTModule
from abc import ABC, abstractmethod


class JDSTAgent(JDSTModule, ABC):
    """
    Base class for an agent. This class inherits from the module base class.
    """

    # Get a prediction:
    @abstractmethod
    def predict(self):
        raise NotImplementedError