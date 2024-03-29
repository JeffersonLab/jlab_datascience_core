from jlab_datascience_toolkit.core.jdst_module import JDSTModule
from abc import ABC, abstractmethod

class JDSTModel(JDSTModule,ABC):
    '''
    Base class for the model. This class inherits from the module base class.
    '''

    # Train the model:
    @abstractmethod
    def train(self):
        raise NotImplementedError

    # Get a prediction:
    @abstractmethod
    def predict(self):
        raise NotImplementedError
    
    # Run a small analysis (e.g. determine ROC-Curve, MSE,...)
    @abstractmethod
    def analysis(self):
        raise NotImplementedError
    