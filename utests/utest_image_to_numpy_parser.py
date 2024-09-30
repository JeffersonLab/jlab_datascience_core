import jlab_datascience_toolkit.data_parsers as parsers
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

class UTestImageToNumpyParser(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestImageToNumpyParser,self).__init__(*args, **kwargs)

        # Get an into:
        print(" ")
        print("****************************************")
        print("*                                      *")
        print("*   Unit Test: Image To Numpy Parser   *")
        print("*                                      *")
        print("****************************************")
        print(" ")

        # First, we download the MNIST data set:
        print("Get MNIST data...")
    
        (x_train,_), _ = tf.keras.datasets.mnist.load_data()

        print("...done!")
        print(" ") 

        # Then randomly pick N images:
        self.n_images = 100

        print("Randomly pick " + str(self.n_images) + " images and convert them to .png files...")

        idx = np.random.randint(0,x_train.shape[0],(self.n_images,))
        images = x_train[idx]

        # Store the images locally as .png files:
        store_name = 'mnist_image'
        # Collect the image names in a list so that we can use them in the parser config:
        self.image_names = []
        #+++++++++++++++++++++
        for i in range(self.n_images):
            current_name = store_name+str(i) + '.png'
            self.image_names.append(current_name)

            fig,ax = plt.subplots()

            ax.imshow(images[i])
            fig.savefig(current_name)

            plt.close(fig)
        #+++++++++++++++++++++

        print("...done!")
        print(" ")

        print("Load parser...")

        # Set up the configuration file for the image to numpy parser:
        parser_cfg = {
            'image_loc': self.image_names, #--> Provide a list of images
        }
        this_file_loc = os.path.dirname(__file__)
        cfg_loc = os.path.join(this_file_loc,'../jlab_datascience_toolkit/cfgs/defaults/image_to_numpy_parser_cfg.yaml')
        # Now get the parser:
        self.parser = parsers.make("ImageToNumpyParser_v0",path_to_cfg=cfg_loc,user_config=parser_cfg)

        # Lets see if we can call the information about this module:
        self.parser.get_info()

        print("...done!")
        print(" ")
    #*****************************************
    
    # Test everything:
    #*****************************************
    def test_drive_image_to_numpy_parser(self):
        # Get the data:
        print("Load data...")

        data = self.parser.load_data()
        
        print("...done!")
        print(" ")
        
        print("Remove .png files...")

        # Delete the .png files so that we do not spam our machine:
        #++++++++++++++++++++
        for i in self.image_names:
            os.remove(i)
        #++++++++++++++++++++

        print("...done!")
        print(" ")

        # We expect the data to have dimension: n_images x height x width x 3
        # We do not care about height, width as it might vary with the settings in imshow. 
        # Thus, we check if we have 4 dimensions and if the 1st and 4th dimension match
        print("Run consistency check...")

        pass_dimension_check = False
        if len(data.shape) == 4 and data.shape[0] == self.n_images and data.shape[3] == 3:
            pass_dimension_check = True

        # If everything is done right, this should turn true:
        self.assertTrue(pass_dimension_check)

        print("...done!")
        print(" ")
    #*****************************************

# Run this file via: python utest_numpy_parser.py
if __name__ == "__main__":
    unittest.main()