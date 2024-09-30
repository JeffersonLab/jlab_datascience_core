import jlab_datascience_toolkit.models as models
import jlab_datascience_toolkit.analyses as analyses
from jlab_datascience_toolkit.utils.get_mnist import get_mnist_data
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

class UTestDataReconstruction(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestDataReconstruction,self).__init__(*args, **kwargs)

        # Get an intro:
        print(" ")
        print("*************************************")
        print("*                                   *")
        print("*   Data Reconstruction Unit-Test   *")
        print("*                                   *")
        print("*************************************")
        print(" ")

         # First,get the MNIST data set offline or online:
        print("Get MNIST data...")
    
        data, _, _, _ = get_mnist_data()

        print("...done!")
        print(" ") 

        # Do some minor preprocessing:
        print("Preprocess data...")

        data = data.reshape((data.shape[0], 28, 28, 1)) / 255.
        self.data = np.where(data > .5, 1.0, 0.0).astype('float32')

        print("...done!")
        print(" ") 
    #*****************************************
    
    # Test drive the model:
    #*****************************************
    def test_data_reconstruction(self):
        # Set the model id:
        model_id = 'KerasCNNAE_v0'
        
        # Store the results of this unit-test somewhere:
        result_loc = 'results_utest_data_reconstruction'
        os.makedirs(result_loc,exist_ok=True)

        # Get the default configuration:
        this_file_loc = os.path.dirname(__file__)
        model_cfg_loc = os.path.join(this_file_loc,'../jlab_datascience_toolkit/cfgs/defaults/keras_cnn_ae_cfg.yaml')

        # And maybe add some user specific values:
        use_conv_latent_layer = False
        model_user_cfg = {
            'image_dimensions':(self.data.shape[1],self.data.shape[2],self.data.shape[3]),
            'n_epochs': 10,
            'dense_architecture':[3,3],
            'dense_activations':['relu']*2,
            'dense_kernel_inits':['he_normal']*2,
            'dense_bias_inits':['he_normal']*2,
            'latent_space_is_2d':use_conv_latent_layer,
            'optimizer':'legacy_adam',
            'early_stopping_monitor':'val_loss',
            'early_stopping_min_delta':0.00005,
            'early_stopping_patience':3,
            'early_stopping_restore_best_weights':True,
        }

        print("Set up model...")

        # Set the model:
        model = models.make(model_id,path_to_cfg=model_cfg_loc,user_config=model_user_cfg)

        # And the model list:
        model_list = model.get_model()

        print("...done!")
        print(" ") 

        # Get id for analysis module:
        ana_id = "DataReconstruction_v0"

        # Specify configuration:
        ana_cfg_loc = os.path.join(this_file_loc,'../jlab_datascience_toolkit/cfgs/defaults/data_reconstruction_cfg.yaml')
        
        data_fraction_to_analyze = 200
        ana_user_cfg = {
            'output_loc': result_loc,
            'analysis_sample_size': 50,
            'n_analysis_samples':data_fraction_to_analyze
        }

        # Load the module:
        print("Load data reconstruction module...")

        analyzer = analyses.make(ana_id,path_to_cfg=ana_cfg_loc,user_config=ana_user_cfg)

        print("...done!")
        print(" ")

        # Run the reconstruction:
        print("Run reconstruction...")

        rec_data = analyzer.run(self.data,model_list)

        print("...done!")
        print(" ")

        # We compare the shapes of the input and reconstructed data... Ideally they should be equal...
        pass_dim_check = False
        if data_fraction_to_analyze > 0:
            if rec_data['x_rec'].shape[0] == rec_data['x_orig'].shape[0] and rec_data['x_rec'].shape[1] == rec_data['x_orig'].shape[1] and rec_data['x_rec'].shape[2] == rec_data['x_orig'].shape[2]:
                pass_dim_check = True

        else:
            if rec_data['x_rec'].shape[0] == self.data.shape[0] and rec_data['x_rec'].shape[1] == self.data.shape[1] and rec_data['x_rec'].shape[2] == self.data.shape[2]:
                pass_dim_check = True
        
        self.assertTrue(pass_dim_check)
    #*****************************************


# Run this file via: python utest_data_reconstruction.py
if __name__ == "__main__":
    unittest.main()