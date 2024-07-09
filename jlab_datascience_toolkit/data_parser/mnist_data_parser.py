from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
from jlab_datascience_toolkit.utils.get_mnist import get_mnist_data
from PIL import Image
import os
import numpy as np
import logging
import inspect
import yaml
from sklearn.utils import shuffle

class MNISTDataParser(JDSTDataParser):
    '''
    Dummy data parser that does not require any specific inputs (e.g. paths or data files) and simply returns the MNIST data (without labels!)
    '''

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        # Set the name specific to this module:
        self.module_name = "image_to_numpy_parser"

        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        self.train_data_percentage = self.config['train_data_percentage']
        self.validation_data_percentage = self.config['validation_data_percentage']
        self.use_labels = self.config['use_labels']

        # Keep track of the labels:
        self.mnist_labels = None
    #*********************************************

    # Check input data which is not necessary here, as this module does not require any:
    #*********************************************
    def check_input_data_type(self):
        pass
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

    # Now load multiple files:
    def load_data(self):
        # Get the mnist data 
        x_train,y_train,x_val,y_val = get_mnist_data()
        
        # Select specific portions:
        n_train = int(self.train_data_percentage*x_train.shape[0])
        n_val = int(self.validation_data_percentage*x_val.shape[0])

        idx_train = np.random.choice(x_train.shape[0],n_train)
        idx_val = np.random.choice(x_val.shape[0],n_val)

        # And combine them to new data:
        new_data = np.concatenate([
            x_train[idx_train],
            x_val[idx_val]
        ],axis=0)
        
        # Use labels if requested by the user:
        if self.use_labels == True:
            logging.info(">>> " + self.module_name + ": Using MNIST labels as well <<<")
            new_labels = np.concatenate([
              y_train[idx_train],
              y_val[idx_val]
            ],axis=0)

            new_data, self.mnist_labels = shuffle(new_data,new_labels)
            return new_data

        return shuffle(new_data)
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