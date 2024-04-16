from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
from PIL import Image
import os
import numpy as np
import logging
import inspect
import yaml

class ImageToNumpyParser(JDSTDataParser):

    """Image to Numpy data parser that reads in strings of file paths to images and returns a single .npy file

    What this module does:
    "i) Read in multiple .png files that are specified in a list of strings
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
        self.module_name = "image_to_numpy_parser"

        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        # Run sanity check(s):
        # i) Make sure that the provide data path(s) are list objects:
        if isinstance(self.config['image_loc'],list) == False:
            logging.error(">>> " + self.module_name +": The data path(s) must be a list object, e.g. data_loc: [path1,path2,...] <<<")
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

    # Load images:
    #*********************************************
    # Load a single image:
    def load_single_image(self,path):
        try:
               img = Image.open(path)
               if self.config['convert_image_mode'] is not None:
                   img = img.convert(self.config['convert_image_mode'])
                
               data = np.array(img).astype(self.config['dtype']) 
               return data
        except:
            logging.exception(f">>> " + self.module_name + ": File {path} does not exist <<<")

    #-----------------------------

    # Now load multiple files:
    def load_data(self):
        try:
            collected_data = []
            #+++++++++++++++++++++
            for path in self.config['image_loc']:
                collected_data.append(np.expand_dims(self.load_single_image(path),axis=self.config['event_axis']))
            #+++++++++++++++++++++

            return np.concatenate(collected_data,axis=self.config['event_axis'])
        except:
            logging.exception(">>> " + self.module_name + ": Please check the provided data path which must be a list. <<<")
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