from abc import ABC, abstractmethod

class JDSTModule(ABC):

    '''
    Base class for any module that is written for the JLab Data Science Toolkit. The functions defined here have to be implemented in 
    any new module that is written
    '''
    
    # Initialize:
    def __init__(self,**kwargs):
        self.module_name = "" # --> Define the name of the module

    # Get module info: Just briefly describe what this module is doing, 
    # what are the inputs and what is returned?
    def get_info(self):
        return NotImplemented
    
    # Load and save configuration files which run  the module:
    def load_config(self):
        return NotImplemented
    
    def save_config(self):
        return NotImplemented
    
    # Load and save for checkpointing (i.e. capture state of module)
    def load(self):
        return NotImplemented
    
    def save(self):
        return NotImplemented