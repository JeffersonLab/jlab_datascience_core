from jlab_datascience_toolkit.core.jdst_module import JDSTModule
from abc import ABC, abstractmethod

class JDSTDataParser(JDSTModule,ABC):

    '''
    Base class for data parsing. This class inherits from the module base class.
    '''
    
    # Load and save the data:
    def load_data(self):
        raise NotImplementedError
    
    def save_data(self):
        raise NotImplementedError