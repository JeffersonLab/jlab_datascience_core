from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
import numpy as np
import yaml
import logging
import inspect
import os

class NumpyParser(JDSTDataParser):
    """Numpy data parser that reads in strings of file paths and returns a single .npy file

    What this module does:
    "i) Read in multiple .npy files that are specified in a list of strings
    ii) Combine single .npy files into one

    Input(s):
    i) Full path to .yaml configuration file 
    ii) Optional: User configuration, i.e. a python dict with additonal / alternative settings

    Output(s):
    i) Single .npy file
    """

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        # Set the name specific to this module:
        self.module_name = "numpy_parser"

        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        # Run sanity check(s):
        # i) Make sure that the provide data path(s) are list objects:
        if isinstance(self.config['data_loc'],list) == False:
            logging.error(">>> " + self.module_name +": The data path(s) must be a list object, e.g. data_loc: [path1,path2,...] <<<")
    #*********************************************

    # Check the input data type --> This module expects a list of strings / file paths:
    #*********************************************
    def check_input_data_type(self,input_data):
        if isinstance(input_data,list) == False:
            logging.error(f">>> {self.module_name}: The input data type {type(input_data)} is not a list. Please correct. Going to returne None <<<")
            return False
        else:
            if len(input_data) > 0:
                return True
            else:
                logging.error(f">>> {self.module_name}: The list of filepaths your provided {input_data} seems to be empty. Please check your configuration. Going to return None <<<")
                return False
    #*********************************************

    # Provide information about this module:
    #*********************************************
    def get_info(self):
        print(inspect.getdoc(self))
    #*********************************************

    # Handle configurations:
    #*********************************************
    # Load the config:
    def load_config(self,path_to_cfg,user_config):
        with open(path_to_cfg, 'r') as file:
            cfg = yaml.safe_load(file)
        
        # Overwrite config with user settings, if provided
        try:
            if bool(user_config):
              #++++++++++++++++++++++++
              for key in user_config:
                cfg[key] = user_config[key]
              #++++++++++++++++++++++++
        except:
            logging.exception(">>> " + self.module_name +": Invalid user config. Please make sure that a dictionary is provided <<<") 

        return cfg
    
    #-----------------------------

    # Store the config:
    def save_config(self,path_to_config):
        with open(path_to_config, 'w') as file:
           yaml.dump(self.config, file)
    #*********************************************
    
    # Load .npy file(s):
    #*********************************************
    # Load a single file:
    def load_single_file(self,path_to_file):
        try:
            return np.load(path_to_file).astype(self.config['dtype'])
        except:
            logging.exception(">>> " + self.module_name + ": File does not exist! <<<")

    #-----------------------------

    # Load multiple files which represent the final data:
    def load_data(self):

        if self.check_input_data_type(self.config['data_loc']) == True:
      
            collected_data = []
            #+++++++++++++++++++++
            for path in self.config['data_loc']:
                collected_data.append(self.load_single_file(path))
            #+++++++++++++++++++++

            return np.concatenate(collected_data,axis=self.config['event_axis'])
        
        return None
    #*********************************************
    
    # Save the data:
    #*********************************************
    def save_data(self,data):
        try:
           os.makedirs(self.config['data_store_loc'],exist_ok=True) 
           np.save(self.config['data_store_loc'],data)
        except:
           logging.exception(">>> " + self.module_name + ": Please provide a valid name for storing the data in .npy format. <<<")
    #*********************************************
    
    # Module checkpointing: Not implemented yet and maybe not 
    # necessary, as we leave these functions blank for now
    #*********************************************
    def load(self):
        return 0

    #-----------------------------

    def save(self):
        return 0
    #*********************************************

    
    

