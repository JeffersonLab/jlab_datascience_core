from jlab_datascience_toolkit.core.jdst_module import JDSTModule
from abc import ABC, abstractmethod

class JDSTModel(JDSTModule,ABC):
    '''
    Base class for the model. This class inherits from the module base class.
    '''

    # Train the model:
    def train(self):
        return NotImplemented

    # Get a prediction:
    def predict(self):
        return NotImplemented
    
    # Run a small analysis (e.g. determine ROC-Curve, MSE,...)
    def analysis(self):
        return NotImplemented
    