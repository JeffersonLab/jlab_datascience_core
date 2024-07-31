from jlab_datascience_toolkit.core.jdst_model import JDSTModel
import jlab_datascience_toolkit.model as models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import inspect
import optuna
import yaml
import logging
import os
import gc

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
 
        # Basic settings:

        # ID of the CNN AE model we wish to tune:
        self.model_id = self.config['model_id']
        # Get the model configuration:
        self.model_cfg_loc = self.config['model_cfg_loc']
        # If there is no location for a model configuration specified, be just go back and load the default one:
        if self.model_cfg_loc == "" or self.model_cfg_loc is None:
            this_file_loc = os.path.dirname(__file__)
            self.model_cfg_loc = os.path.join(this_file_loc,'../cfgs/defaults/keras_cnn_ae_cfg.yaml')
        
        # Load the config:
        with open(self.model_cfg_loc, 'r') as cfg:
            self.model_config = yaml.safe_load(cfg)
        
        # Get the iamge dimensions right:
        self.model_config['image_dimensions'] = self.config['image_dimensions']
        # Add max pooling:
        self.model_config['max_pooling'] = self.config['max_pooling']
        # Optimizer:
        self.model_config['optimizer'] = self.config['optimizer']

        
        # Kernel size and strides:
        self.kernel_size = self.config['kernel_size']
        self.stride = self.config['stride']

        # HPO specific settings:
        self.n_hpo_trials = self.config['n_hpo_trials']
        self.n_epochs_per_trial = self.config['n_epochs_per_trial']
        self.batch_size_per_trial = self.config['batch_size_per_trial']
        self.validation_split_per_trial = self.config['validation_split_per_trial']
        self.verbosity_per_trial = self.config['verbosity_per_trial']
        self.hpo_objective_fn = self.config['hpo_objective_fn']
        self.hpo_objective_direction = self.config['hpo_objective_direction']
        self.hpo_result_folder = self.config['hpo_result_folder']
        self.hpo_study_name = self.config['hpo_study_name']
        self.hpo_param_importance = self.config['hpo_param_importance']
 
        # Training of the final model:
        self.n_epochs = self.config['n_epochs']
        self.batch_size = self.config['batch_size']
        self.validation_split = self.config['validation_split']
        self.verbosity = self.config['verbosity']

        # Weight and bias initialization:
        self.conv_kernel_initialization = self.config['conv_kernel_initialization']
        self.conv_bias_initialization = self.config['conv_bias_initialization']
        self.dense_kernel_initialization = self.config['dense_kernel_initialization']
        self.dense_bias_initialization = self.config['dense_bias_initialization']


        # Tuneabale parameters:

        # Conv. Architecture:
        self.max_n_conv_layers = self.config['max_n_conv_layers']
        self.step_n_conv_layers = self.config['step_n_conv_layers']
        self.min_conv_filters = self.config['min_conv_filters']
        self.max_conv_filters = self.config['max_conv_filters']
        self.step_conv_filters = self.config['step_conv_filters']
        # Dense Architecture:
        self.max_n_dense_layers = self.config['max_n_dense_layers']
        self.step_n_dense_layers = self.config['step_n_dense_layers']
        self.min_dense_units = self.config['min_dense_units']
        self.max_dense_units = self.config['max_dense_units']
        self.step_dense_units = self.config['step_dense_units']
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

        # Set up the optimizer:
        # Collect results:
        os.makedirs(self.hpo_result_folder,exist_ok=True) 
        # Write this config to file:
        self.save_config(self.hpo_result_folder+"/hpo_configuration.yaml")

        # Preparation for objective scan:
        self.score = 1E99
        self.hpo_data = None
        self.model = None
        self.maximize_objective = False
        if self.hpo_objective_direction.lower() == "maximize":
            self.score = -1E99
            self.maximize_objective = True
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
    
    # Probe the hyper parameter space --> We overwrite the original model configuration and then load the new model
    #*********************************************
    # Adjust weight / bias intitialization:
    def initialize_layers(self,activations,kernel_init_str,bias_init_str):
        kernel_inits = []
        bias_inits = []

        if len(activations) > 0:
           #++++++++++++++++++++
           for act in activations:
               if act.lower() == "relu" or act.lower() == "leaky_relu":
                   kernel_inits.append("he_"+kernel_init_str)
               elif act.lower() == "selu":
                   kernel_inits.append("lecun_"+kernel_init_str)
               else:
                   kernel_inits.append("glorot_"+kernel_init_str)

               bias_inits.append(bias_init_str) 
           #++++++++++++++++++++  

        return kernel_inits, bias_inits

    #-----------------------------

    # Set hyper parameters for a give trial:
    def probe_hp_space(self,trial):
        # Record the trial number:
        current_trial = trial.number

        # Copy original model config file, just to be on the safe side:
        current_model_cfg = self.model_config.copy()

        # Update hyper parameters
        current_model_cfg['trial'] = current_trial

        # Conv. architecture:
        if self.step_n_conv_layers > 0 and self.max_n_conv_layers > 1:
            n_conv_layers = trial.suggest_int("n_conv_layers",1,self.max_n_conv_layers,step=self.step_n_conv_layers)

        if self.max_conv_filters > 1 and self.min_conv_filters > 0 and self.step_conv_filters > 0:
           new_conv_architecture = []
           n_filters_prev = 0
           #+++++++++++++++++++++++++
           for k in range(n_conv_layers):
              n_filters = trial.suggest_int(f'n_filters_layer{k}',n_filters_prev+self.min_conv_filters,n_filters_prev+self.max_conv_filters,step=self.step_conv_filters)
              new_conv_architecture.append(n_filters)

              n_filters_prev = n_filters
           #+++++++++++++++++++++++++
           current_model_cfg['conv_architecture'] = new_conv_architecture
           current_model_cfg['kernel_sizes'] = [self.kernel_size] * len(current_model_cfg['conv_architecture'])
           current_model_cfg['strides'] = [self.stride] * len(current_model_cfg['conv_architecture'])

        # Dense architecture:
        if self.step_n_dense_layers > 0 and self.max_n_dense_layers > 1:
            n_dense_layers = trial.suggest_int("n_dense_layers",1,self.max_n_dense_layers,step=self.step_n_dense_layers)
        
        if self.max_dense_units > 1 and self.min_dense_units > 0 and self.step_dense_units > 0:
            new_dense_architecture = []
            n_units_prev = 0
            #+++++++++++++++++++++++++
            for d in range(n_dense_layers):
               n_units = trial.suggest_int(f'n_units_layer{d}',n_units_prev+self.min_dense_units,n_units_prev+self.max_dense_units,step=self.step_dense_units)
               new_dense_architecture.append(n_units)

               n_units_prev = n_units
            #+++++++++++++++++++++++++
            current_model_cfg['dense_architecture'] = new_dense_architecture
            # Need to update the number of dense layers:
            n_dense_layers = len(current_model_cfg['dense_architecture'])
        

        # Activations:
        if len(self.conv_activation) > 1:
            conv_act = trial.suggest_categorical("conv_activation",self.conv_activation)
            current_model_cfg['conv_activations'] = [conv_act] * len(current_model_cfg['conv_architecture'])
        elif len(self.conv_activation) == 1:
            current_model_cfg['conv_activations'] = self.conv_activation * len(current_model_cfg['conv_architecture'])

        if len(self.dense_activation) > 1 and n_dense_layers > 0:
            dense_act = trial.suggest_categorical("dense_activation",self.conv_activation)
            current_model_cfg['dense_activations'] = [dense_act] * len(current_model_cfg['dense_architecture'])
        elif len(self.dense_activation) == 1:
            current_model_cfg['dense_activations'] = self.dense_activation * len(current_model_cfg['dense_architecture'])

        # Weight and bias initialization:
        conv_kernel_init, conv_bias_init = self.initialize_layers(current_model_cfg['conv_activations'],self.conv_kernel_initialization,self.conv_bias_initialization)
        dense_kernel_init, dense_bias_init = self.initialize_layers(current_model_cfg['dense_activations'],self.dense_kernel_initialization,self.dense_bias_initialization)
      
        current_model_cfg['conv_kernel_inits'] = conv_kernel_init
        current_model_cfg['conv_bias_inits'] = conv_bias_init
        current_model_cfg['dense_kernel_inits'] = dense_kernel_init
        current_model_cfg['dense_bias_inits'] = dense_bias_init

        # Latene space:
        if self.min_latent_dim > 0 and self.max_latent_dim > 1 and self.step_latent_dim > 0:
           current_model_cfg['latent_dim'] = trial.suggest_int('latent_dim',self.min_latent_dim,self.max_latent_dim,step=self.step_latent_dim)

        # Learning rate:
        if self.min_learning_rate > 0.0 and self.max_learning_rate > self.min_learning_rate:
            current_model_cfg['learning_rate'] = trial.suggest_float("learning_rate",self.min_learning_rate,self.max_learning_rate,log=True)

        return current_model_cfg
    #*********************************************
    
    # Objective:
    #*********************************************
    def objective(self,trial):
        # Clear the memory:
        tf.keras.backend.clear_session()

        # Probe hyper paramteres and get model settings:
        current_settings = self.probe_hp_space(trial)

        # Create a new model:
        current_model = models.make(self.model_id,path_to_cfg=self.model_cfg_loc,user_config=current_settings)

        # Train the model for a bit, to extract the objective:
        current_results = current_model.train(
            x=self.hpo_data,
            n_epochs=self.n_epochs_per_trial,
            batch_size=self.batch_size_per_trial,
            validation_split=self.validation_split_per_trial,
            verbosity=self.verbosity_per_trial
        )

        # Retreive objective:
        objective_score = None
        if self.hpo_objective_fn in current_results:
           objective_score = current_results[self.hpo_objective_fn][-1]
        else:
           objective_score = current_results['loss'][-1]

        # Make sure we do not collect garbage
        gc.collect()
        
        # Update the objective score:
        if np.isnan(objective_score) or np.isposinf(objective_score):
            objective_score  = 1E99
        
        if np.isneginf(objective_score):
            objective_score = -1E99 
        
        # Compare the current performance to the previous one and then select the 'best' model:
        if self.maximize_objective == True:
             if objective_score > self.score:
                self.model = current_model
        else:
             if objective_score < self.score:
                self.model = current_model
                
        del current_model
        del current_settings
        
        return objective_score 
    #*********************************************
    
    # Visualize HP performance:
    #*********************************************
    def visualize_search(self,optuna_study):
        # Optimization history:
        optuna.visualization.matplotlib.plot_optimization_history(optuna_study)
        plt.gcf().set_size_inches(20,7)
        plt.savefig(self.hpo_result_folder+"/optimization_history.png")
        plt.close()
        # Parameter Importance:
        optuna.visualization.matplotlib.plot_param_importances(optuna_study)
        plt.gcf().set_size_inches(15,7)
        plt.savefig(self.hpo_result_folder+"/hp_importance.png")
        plt.close()
        # Parallel plot:
        optuna.visualization.matplotlib.plot_parallel_coordinate(optuna_study, params=self.hpo_param_importance)
        plt.gcf().set_size_inches(15,7)
        plt.savefig(self.hpo_result_folder+"/hp_parallel.png")
        plt.close()
    #*********************************************
    
    # Get the predicion:
    #*********************************************
    def predict(self,x,to_numpy=True):
        return self.model.predict(x,to_numpy)
    #*********************************************
    
    # Run the HP search and the final training of the best model:
    #*********************************************
    def train(self,x):
        # Define a study:
        study = optuna.create_study(direction=self.hpo_objective_direction,study_name=self.hpo_study_name)
        
        self.hpo_data = x
        # Run the optimization:
        study.optimize(self.objective,n_trials=self.n_hpo_trials,gc_after_trial=True)

        # And visualize everything:
        self.visualize_search(study)

        # Write the 'final' configuration to file:
        cfg_loc = self.hpo_result_folder + "/best_model_settings"
        self.model.save_config(cfg_loc)

        # Finally, train the 'best' model for a few more epochs:
        results = self.model.train(
            x=x,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbosity=self.verbosity
        )
        
        # Store the model itself:
        self.save(self.hpo_result_folder)

        return results
    #*********************************************

    # Store / load  the network:
    #****************************
    # Save the entire model:
    def save(self,model_loc):
        self.model.save(model_loc)

    #----------------

    def load(self):
        pass
    #****************************


    # Get the encoder / decoder models themselves:
    #*********************************************
    def get_model(self,x=None):
        return self.model.get_model()
    #*********************************************

