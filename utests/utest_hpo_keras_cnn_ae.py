import jlab_datascience_toolkit.models as models
from jlab_datascience_toolkit.utils.get_mnist import get_mnist_data
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

class UTestHPOKerasCNNAE(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestHPOKerasCNNAE,self).__init__(*args, **kwargs)

        # Get an intro:
        print(" ")
        print("**********************************")
        print("*                                *")
        print("*   HPO Keras CNN AE Unit-Test   *")
        print("*                                *")
        print("**********************************")
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
    def test_drive_model(self):
        # Set the model id:
        model_id = 'HPOKerasCNNAE_v0'

        # Store the results of this unit-test somewhere:
        result_loc = 'results_utest_hpo_keras_cnn_ae'
        os.makedirs(result_loc,exist_ok=True)

        # Get the default configuration:
        this_file_loc = os.path.dirname(__file__)
        cfg_loc = os.path.join(this_file_loc,'../jlab_datascience_toolkit/cfgs/defaults/hpo_keras_cnn_ae_cfg.yaml')

        # And maybe add some user specific values:
        use_conv_latent_layer = False
        user_cfg = {
            'image_dimensions':(self.data.shape[1],self.data.shape[2],self.data.shape[3]),
            'hpo_result_folder':result_loc,
            'max_pooling':[2,2],
            'optimizer': 'legacy_adam',
            'n_hpo_trials': 10,
            'n_epochs_per_trial': 10,
            'n_epochs': 20
        }

        print("Set up model...")

        # Set the model:
        hpo_model = models.make(model_id,path_to_cfg=cfg_loc,user_config=user_cfg)

        print("...done!")
        print(" ") 

        # Do a short training:
        print("Run short training...")

        loss_dict = hpo_model.train(self.data)

        print("...done!")
        print(" ") 

        print("Test model response...")

        preds = hpo_model.predict(self.data[:6],True)
        rec_data = preds['x_rec']
        latent_data = preds['z_model']

        print("...done!")
        print(" ") 

        print("Visualize and store results...")
        plt.rcParams.update({'font.size':20})
        
        # Reconstruction:
        figr,axr = plt.subplots(2,6,figsize=(17,8),sharex=True,sharey=True)

        #++++++++++++++++++
        for i in range(6):
            axr[0,i].imshow(self.data[i])
            axr[1,i].imshow(rec_data[i])
        #++++++++++++++++++

        figr.savefig(result_loc+"/reconstructed_data.png")
        plt.close(figr)
        
        # Latent dimension:
        if use_conv_latent_layer == True:
            figl,axl = plt.subplots(1,6,figsize=(17,8))

            #++++++++++++++++++
            for i in range(6):
               axl[i].imshow(latent_data[i])
            #++++++++++++++++++
               
            figl.savefig(result_loc+"/latent_features.png")
            plt.close(figl)

        # Training curves:
        fig,ax = plt.subplots(figsize=(12,8))

        ax.plot(loss_dict['loss'],linewidth=3.0,label='Training')
        ax.plot(loss_dict['val_loss'],linewidth=3.0,label='Validation')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()

        fig.savefig(result_loc+"/learning_curves.png")
        plt.close(fig)

        print("...done!")
        print(" ")

        # The loss dict should have two losses: one for training and one for validation. Each should
        # be lists with n_epochs elements:
        pass_dim_test = False
        if len(loss_dict['loss']) == len(loss_dict['val_loss']):
            pass_dim_test = True

        self.assertTrue(pass_dim_test)
    #*****************************************


# Run this file via: python utest_hpo_keras_cnn_ae.py
if __name__ == "__main__":
    unittest.main()