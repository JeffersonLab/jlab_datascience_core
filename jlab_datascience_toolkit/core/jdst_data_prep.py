from jlab_datascience_toolkit.core.jdst_module import JDSTModule
from abc import ABC, abstractmethod

class JDSTDataPrep(JDSTModule,ABC):

    '''
    Base class for data preparation. This class inherits from the module base class.
    '''

    # Save the data, if required
    # This might be helpful, if the underlying data preperation is a computational intensive operation
    # And we want to avoid calling it multiple times. Thus, we just store the data after preparation:
    def save_data(self):
        return NotImplemented
    
    # Run the data preparation:
    def run(self):
        return NotImplemented

    # Reverse the data preparation (if possible):
    def reverse(self):
        return NotImplemented