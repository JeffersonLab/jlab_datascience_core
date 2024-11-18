from jlab_datascience_toolkit.core.jdst_module import JDSTModule
from abc import ABC, abstractmethod


class JDSTTrainer(JDSTModule, ABC):
    """
    Base class for the Trainer. This class inherits from the module base class.
    """

    # Get a prediction:
    @abstractmethod
    def fit(self):
        raise NotImplementedError
