from jlab_datascience_toolkit.cores.jdst_module import JDSTModule
from abc import ABC, abstractmethod


class JDSTAnalysis(JDSTModule, ABC):
    """
    Base class for the post-training analysis. This class inherits from the module base class.
    """

    # Run the analysis:
    @abstractmethod
    def run(self):
        raise NotImplementedError
