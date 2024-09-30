from jlab_datascience_toolkit.core.jdst_analysis import JDSTAnalysis
import tensorflow as tf
import numpy as np
import os
import inspect
import yaml
import logging

class DataReconstruction(JDSTAnalysis):
    '''
    Simple module that passes input data x through a model:
    
    x_rec = model(x)

    where model can be a (variational) Autoencoder, U-Net, Diffusion model,...
    The data here is processed via the tf.dataset system, in order to efficiently handle large data sets.

    Input(s):
    i) Numpy arrays / images
    ii) A trained model

    Output(s):
    i) Dictionary with reconstructed images and (optional) original images
    '''

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        # Define the module and module name:
        self.module_name = "data_reconstruction"

         # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])
        
        # General settings:
        self.output_loc = self.config['output_loc']
        self.data_store_name = self.config['data_store_name'] 

        # Data processing settings:
        self.buffer_size = self.config['buffer_size']
        self.n_analysis_samples = self.config['n_analysis_samples']
        self.analysis_sample_size = self.config['analysis_sample_size']

        # Get names of the data:
        self.data_names = self.config['data_names']
        self.record_original_data = self.config['record_original_data']
   
        self.store_data = False
        if self.output_loc is not None and self.output_loc.lower() != "":
           self.store_data = True
           os.makedirs(self.output_loc,exist_ok=True)
    #*********************************************

    # Check the input data type:
    #*********************************************
    def check_input_data_type(self,x=None,model_list=[]):
       
        if isinstance(x,np.ndarray) and isinstance(model_list,list):
                pass_model_type_check = False
                if len(model_list) > 0:
                    pass_model_type_check = True
                    #+++++++++++++++
                    for m in model_list:
                       if isinstance(m,tf.keras.Model) == False:
                           pass_model_type_check = False
                    #+++++++++++++++

                return pass_model_type_check
        else:
            logging.error(f">>> {self.module_name}: The provided data does not match the requirements. The first argument has to be a numpy array, Whereas the second argument should be a non-empty list with tf.keras.Model. Going to return None. <<<")
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
    
    # Reconstruct the data:
    #*********************************************
    # First, we need a model prediction:
    def get_model_predictions(self,x,model_list):
        # Go through all elements within the model list and collect the predictions
        x_in = x
        #++++++++++++++++++
        for model in model_list:
            x_out = model.predict_on_batch(x_in)
            x_in = x_out
        #++++++++++++++++++

        return x_out
    
    #------------------------------
    
    # Now run the reconstruction:
    def reconstruct_data(self,x,model_list):
        # First, we need to create a tf data set which shows the beauty of this method:

        # Provide the option to only analyze a part of the initial data:
        n_ana_samples = x.shape[0]
        if self.n_analysis_samples > 0:
            n_ana_samples = self.n_analysis_samples

            # If we only analyze a fraction of the data, we need to record to original data as well:
            self.record_original_data = True

        tf_data = tf.data.Dataset.from_tensor_slices(x).shuffle(buffer_size=self.buffer_size).take(n_ana_samples).batch(self.analysis_sample_size)

        # Second, make sure that we have a model list:
        if type(model_list) != list:
             model_list = [model_list] 

        # Third, make some predictions: 
        predictions = []
        inputs = []
        #++++++++++++++++++++++
        for sample in tf_data:
            # Get the prediction:
            current_pred = self.get_model_predictions(sample,model_list)
            predictions.append(current_pred)

            if self.record_original_data == True:
                inputs.append(sample)
        #++++++++++++++++++++++
        
        # Record everything:
        result_dict = {}
        result_dict[self.data_names[1]] = np.concatenate(predictions,axis=0)

        if self.record_original_data == True:
            result_dict[self.data_names[0]] = np.concatenate(inputs,axis=0)
        else:
            result_dict[self.data_names[0]] = None
    
        return result_dict
    #*********************************************

    # Run the analysis:
    #*********************************************
    def run(self,x,model_list):
        # Run type check:
        if self.check_input_data_type(x,model_list):
           results = self.reconstruct_data(x,model_list)

           if self.store_data:
              np.save(self.output_loc+"/"+self.data_store_name+".npy",np.array(results,dtype=object))

           return results
        
        else:
            return None
    #*********************************************

    # Save and load are not active here:
    #****************************
    def save(self):
        pass

    def load(self):
        pass
    #****************************

