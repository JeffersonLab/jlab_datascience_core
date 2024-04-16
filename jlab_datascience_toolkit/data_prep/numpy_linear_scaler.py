from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep
import numpy as np
import yaml
import inspect
import logging
import os

class NumpyLinearScaler(JDSTDataPrep):
    """Simplified linear scaler

    What this module does:
    "i) Apply the transformation: A * X + B, where A, B are constants and X is a numpy array / image

    Input(s):
    i) Scale A
    ii) Offset B
    iii) dtype (int,float,etc.) for the scaled data
    iv) dtype (int,float,etc.) for the reverse scaled data

    Output(s):
    i) Scaled image
    """

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        # Set the name specific to this module:
        self.module_name = "numpy_linear_scaler"
        
        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])
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

    # Run a type chec:
    #*********************************************
    def type_check(self,data):
        if isinstance(data,np.ndarray) == False:
            logging.error(">>> " + self.module_name + ": Data is not a numpy array <<<")
            return False
        
        return True
    #*********************************************
    
    # Run and reverse the scaling: 
    #*********************************************
    # Scale:
    def run(self,data):
        if self.type_check(data) == True:
           return (data * self.config['A'] + self.config['B']).astype(self.config['run_dtype'])
        
    #-----------------------------

    # Undo scaling:
    def reverse(self,data):
        if self.type_check(data) == True:
            reversed_data = (data - self.config['B']) / self.config['A']
            return reversed_data.astype(self.config['reverse_dtype'])
    #*********************************************

     # Save the data:
    #*********************************************
    def save_data(self,data):
        try:
           os.makedirs(self.config['data_store_loc'],exist_ok=True)
           np.save(self.config['data_store_loc'],data)
        except:
           logging.exception(">>> " + self.module_name + ": Please provide a valid name for storing the transformed .npy data <<<")
    #*********************************************

    # Module checkpointing: Save and load parameters that are important to this scaler:
    #*********************************************
    def load(self):
        store_name = self.config['store_loc']
        A = np.load(store_name+"/numpy_linear_scaler_A.npy")
        B = np.load(store_name+"/numpy_linear_scaler_B.npy")
        return {
            'A':A,
            'B':B
        }
    
    #-----------------------------
    
    def save(self):
        store_name = self.config['store_loc']
        os.makedirs(store_name,exist_ok=True)

        np.save(store_name+"/numpy_linear_scaler_A.npy",self.config['A'])
        np.save(store_name+"/numpy_linear_scaler_B.npy",self.config['B'])
    #*********************************************
