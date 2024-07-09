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
    @abstractmethod
    def get_info(self):
        raise NotImplementedError
    
    # Request that every module runs a type check on the input data
    # This helps to ensure that we can faster identify if certain modules can not be combined 
    # e.g. a pytorch module with a tensorflow module, or modules that simply expect different input data types
    @abstractmethod
    def check_input_data_type(self):
        return NotImplementedError

    # Load and save configuration files which run the module:
    @abstractmethod
    def load_config(self):
        raise NotImplementedError
    
    @abstractmethod
    def save_config(self):
        raise NotImplementedError
    
    # Load and save for checkpointing (i.e. capture state of module)
    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError