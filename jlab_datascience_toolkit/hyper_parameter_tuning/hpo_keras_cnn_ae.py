from jlab_datascience_toolkit.core.jdst_model import JDSTModel
import inspect
import optuna
import yaml
import logging

class HPOKerasCNNAE(JDSTModel):
    '''
    Hyper parameter optimization class for a keras CNN AE.
    '''

    # Initialize:
    #****************************
    def __init__(self,path_to_cfg,user_config={}):
        # Set the name specific to this module:
        self.module_name = "keras_cnn_ae"

        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        # Retreive settings from configuration:
        precision = self.config['precision']

        # Tunabale parameters:

        # Conv. Architecture:
        self.max_n_conv_layers = self.config['max_n_conv_layers']
        self.step_n_conv_layers = self.config['step_n_conv_layers']
        self.min_conv_filters = self.config['min_conv_filters']
        self.max_conv_filters = self.config['max_conv_filters']
        self.step_conv_filters = self.config['step_conv_filters']
        # Dense Architecture:
        self.max_n_dense_layers = self.config['max_n_dense_layers']
        self.step_n_dense_layers = self.config['step_n_dense_layers']
        self.min_dense_units = self.config['n_min_dense_units']
        self.max_dense_units = self.config['n_max_dense_units']
        self.step_dense_units = self.config['n_step_dense_units']
        # Latent space:
        self.min_latent_dim = self.config['min_latent_dim']
        self.max_latent_dim = self.config['max_latent_dim']
        self.step_latent_dim = self.config['step_latent_dim']
        # Learning rate:
        self.max_learning_rate = self.config['max_learning_rate']
        self.min_learning_rate = self.config['min_learning_rate']
        # Activation functions:
        self.conv_activation = self.config['conv_activation']
        self.dense_activation = self.config['dense_activation']
    #****************************

    # Pass on the data type check for now, as tensorflow allows a variety of data types
    # see here: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    #****************************
    def check_input_data_type(self):
        pass
    #****************************

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
    
    # Probe the hyper parameter space
    #*********************************************
    def probe_hp_space(self,trial,original_model_cfg):
        # Record the trial number:
        current_trial = trial.number

        # Update hyper parameters
        new_hp = {
            'trial': current_trial
        }

        # Conv. architecture:
        n_conv_layers = len(original_model_cfg['conv_architecture'])
        if self.step_n_conv_layers > 0 and self.max_n_conv_layers > 1:
            n_conv_layers = trial.suggest_int("n_conv_layers",1,self.max_n_conv_layers,step=self.step_n_conv_layers)

        conv_architecture = original_model_cfg['conv_architecture']
        if self.max_conv_filters > 1 and self.min_conv_filters > 0 and self.step_conv_filters > 0:
           conv_architecture = []
           n_filters_prev = 0
           #+++++++++++++++++++++++++
           for k in range(n_conv_layers):
              n_filters = trial.suggest_int(f'n_filters_layer{k}',n_filters_prev+self.min_conv_filters,n_filters_prev+self.max_conv_filters,step=self.step_conv_filters)
              conv_architecture.append(n_filters)

              n_filters_prev = n_filters
           #+++++++++++++++++++++++++

        new_hp['conv_architecture'] = conv_architecture

        # Dense architecture:
        n_dense_layers = len(original_model_cfg['dense_architecture'])
        if self.step_n_dense_layers > 0 and self.max_n_dense_layers > 1:
            n_dense_layers = trial.suggest_int("n_dense_layers",1,self.max_n_dense_layers,step=self.step_n_dense_layers)
        
        dense_architecture = original_model_cfg['dense_architecture']
        if self.max_dense_units > 1 and self.min_dense_units > 0 and self.step_dense_units > 0:
            dense_architecture = []
            n_units_prev = 0
            #+++++++++++++++++++++++++
            for d in range(n_dense_layers):
               n_units = trial.suggest_int(f'n_units_layer{d}',n_units_prev+self.min_dense_units,n_units_prev+self.max_dense_units,step=self.step_dense_units)
               dense_architecture.append(n_units)

               n_units_prev = n_units
            #+++++++++++++++++++++++++

        new_hp['dense_architecture'] = dense_architecture

        # Activations:
        if len(self.conv_activation) > 1:
            conv_act = trial.suggest_categorical("conv_activation",self.conv_activation)
            new_hp['conv_activations'] = [conv_act] * n_conv_layers
        elif len(self.conv_activation) == 1:
            new_hp['conv_activations'] = self.conv_activation * n_conv_layers

        if len(self.dense_activation) > 1 and n_dense_layers > 0:
            dense_act = trial.suggest_categorical("dense_activation",self.conv_activation)
            new_hp['dense_activations'] = [dense_act] * n_dense_layers
        elif len(self.dense_activation) == 1:
            new_hp['dense_activations'] = self.dense_activation * n_dense_layers

        # Latene space:
        latent_dim = original_model_cfg['latent_dim']
        if self.min_latent_dim > 0 and self.max_latent_dim > 1 and self.step_latent_dim > 0:
           latent_dim = trial.suggest_int('latent_dim',self.min_latent_dim,self.max_latent_dim,step=self.step_latent_dim)
        
        new_hp['latent_dim'] = latent_dim

        # Learning rate:
        lr = original_model_cfg['learning_rate']
        if self.min_learning_rate > 0.0 and self.max_learning_rate > self.min_learning_rate:
            lr = trial.suggest_float(self.min_learning_rate,self.max_learning_rate,log=True)
        
        new_hp['learning_rate'] = lr
        
        return new_hp
    #*********************************************

