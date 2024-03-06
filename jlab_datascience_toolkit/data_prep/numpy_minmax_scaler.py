from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import yaml
import os

class NumpyMinMaxScaler(JDSTDataPrep):

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        # Set the name specific to this module:
        self.module_name = "numpy_minmax_scaler"
        
        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        # Set up the scaler:
        try:
            self.scaler = MinMaxScaler(self.config['feature_range'])
        except:
            logging.exception(">>> " + self.module_name + f": Invalid feature range: {self.config['feature_range']}. Must provide a tuple. <<<")
    #*********************************************

    # Provide information about this module:
    #*********************************************
    def get_info(self):
        print("  ")
        print("***   Info: NumpyMinMaxScaler   ***")
        print("Input(s):")
        print("i) Full path to .yaml configuration file ") 
        print("ii) Optional: User configuration, i.e. a python dict with additonal / alternative settings")
        print("iii) Numpy data")
        print("What this module does:")
        print("i) Scale input data with respect to a specified range")
        print("ii) Optional: reverse the scaling")
        print("Output(s):")
        print("i) Scaled .npy data")
        print("ii) Optional: unscaled .npy data")
        print("Note(s):")
        print("i) The scaler will (by default) be fitted to the data and the transform it. To disable the fitting, do: run(data,disable_fit=True)")
        print("***   Info: NumpyMinMaxScaler   ***")
        print("  ")
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
    def run(self,data,disable_fit=False):
        # Check if the data-type is a numpy array:
        if self.type_check(data):

           # Do not re-calibrate the scaler, if a fit has already been done:
           if disable_fit == True:
               return self.scaler.transform(data)

           return self.scaler.fit_transform(data)
        
    #-----------------------------

    # Undo the scaling:
    def reverse(self,data):
        # Run a type check:
        if self.type_check(data):
           return self.scaler.inverse_transform(data)
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
        scaler_min = np.load(store_name+"/numpy_minmax_scaler_min.npy")
        scaler_scale = np.load(store_name+"/numpy_minmax_scaler_scale.npy")
        scaler_data_min = np.load(store_name+"/numpy_minmax_scaler_data_min.npy")
        scaler_data_max = np.load(store_name+"/numpy_minmax_scaler_data_max.npy")

        return {
            'min': scaler_min,
            'scale': scaler_scale,
            'data_min':scaler_data_min,
            'data_max':scaler_data_max
        }
    #-----------------------------
    
    def save(self):
        store_name = self.config['store_loc']
        os.makedirs(store_name,exist_ok=True)

        np.save(store_name+"/numpy_minmax_scaler_min.npy",self.scaler.min_)
        np.save(store_name+"/numpy_minmax_scaler_scale.npy",self.scaler.scale_)
        np.save(store_name+"/numpy_minmax_scaler_data_min.npy",self.scaler.data_min_)
        np.save(store_name+"/numpy_minmax_scaler_data_max.npy",self.scaler.data_max_)
    #*********************************************

    