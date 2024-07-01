import jlab_datascience_toolkit.data_parser as parsers
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os

class UTestMNISTDataParser(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestMNISTDataParser,self).__init__(*args, **kwargs)
        
        # Get an into:
        print(" ")
        print("***********************************")
        print("*                                 *")
        print("*   MNIST Data Parser Unit-Test   *")
        print("*                                 *")
        print("***********************************")
        print(" ")
    #*****************************************
    
    # test the parser:
    #*****************************************
    def test_mnist_data_parser(self):
        print("Load MNIST data parser...")
        
        # Module name:
        module_id = "MNISTDataParser_v0"

        # Location for the default settings:
        this_file_loc = os.path.dirname(__file__)
        cfg_loc = os.path.join(this_file_loc,'../jlab_datascience_toolkit/cfgs/defaults/mnist_data_parser_cfg.yaml')

        # User settings:
        user_cfg = {
            'train_data_percentage': 0.75,
            'test_data_percentage': 0.1,
            'use_labels':True
        }

        mnist_parser = parsers.make(module_id,path_to_cfg=cfg_loc,user_config=user_cfg)
        
        print("...done!")
        print(" ")
        
        # Get the data:
        print("Parse MNIST data...")
        
        mnist_data = mnist_parser.load_data()

        print("...done!")
        print(" ")

        # We simply test if the 0-dim of the returned data and the stored labels are the same
        # --> They should by construction...
        print("Run small sanity check...")
        
        n_data = mnist_data.shape[0]
        n_labels = mnist_parser.mnist_labels.shape[0]

        pass_dim_check = False
        if n_data == n_labels:
            pass_dim_check = True

        print("...done!")
        print(" ")
        
        print("Generate plots for consistency checks..")
        # Create some plots that show that this thing is working:
        test_labels = [0,2,7]
        n_acc = 3

        fig,ax = plt.subplots(n_acc,n_acc,figsize=(18,8))
        counter = 0
        #+++++++++++++++++++++++
        for label in test_labels:
           cond = (mnist_parser.mnist_labels == label)
           current_data = mnist_data[cond]

           idx_acc = np.random.choice(current_data.shape[0],n_acc)
           acc_data = current_data[idx_acc]
           
           ax[counter,0].set_title('Label = ' + str(label))
           ax[counter,0].imshow(acc_data[0])
           ax[counter,0].set_axis_off()
           
           ax[counter,1].set_title('Label = ' + str(label))
           ax[counter,1].imshow(acc_data[1])
           ax[counter,1].set_axis_off()
           
           ax[counter,2].set_title('Label = ' + str(label))
           ax[counter,2].imshow(acc_data[2])
           ax[counter,2].set_axis_off()

           counter += 1
        #+++++++++++++++++++++++

        output_loc = 'results_utest_mnist_data_parser'
        os.makedirs(output_loc,exist_ok=True)

        fig.savefig(output_loc+"/mnist_data_plots.png")
        plt.close(fig)
        
        print("...done! Have a great day!")
        print(" ")

        self.assertTrue(pass_dim_check)
    #*****************************************


# Run this file via: python utest_mnist_data_parser.py
if __name__ == "__main__":
    unittest.main()