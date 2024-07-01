import jlab_datascience_toolkit.analysis as analyses
from jlab_datascience_toolkit.utils.get_mnist import get_mnist_data
import numpy as np
import unittest
import matplotlib.pyplot as plt
import os

class UTestResidualAnalyzer(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestResidualAnalyzer,self).__init__(*args, **kwargs)
        
        # Get an intro:
        print(" ")
        print("***********************************")
        print("*                                 *")
        print("*   Residual Analyzer Unit-Test   *")
        print("*                                 *")
        print("***********************************")
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
    
    # Test everything:
    #*****************************************
    def test_drive_analyzer(self):
        # Create test data:
        print("Create test data...")
        
        # Do not use the entire MNIST data:
        idx = np.random.choice(self.data.shape[0],20)
        acc_data = self.data[idx]

        # Smearing factor:
        f_smear = 5.0
        data_smeared = acc_data * np.random.normal(loc=1.0,scale=f_smear,size=acc_data.shape)

        test_data = {
            'x_real': acc_data,
            'x_rec': data_smeared
        }

        print("...done!")
        print(" ") 

        # Load the analyzer:
        print("Load analyzer module...")

        # We need the name:
        module_id = "ResidualAnalyzer_v0"
        
        # Get the path to the default configuration:
        this_file_loc = os.path.dirname(__file__)
        cfg_loc = os.path.join(this_file_loc,'../jlab_datascience_toolkit/cfgs/defaults/residual_analyzer_cfg.yaml')

        # Specify the name of the output file:
        user_cfg = {
            'output_loc': 'results_utest_residual_analyzer'
        }
        
        analyzer = analyses.make(module_id,path_to_cfg=cfg_loc,user_config=user_cfg)

        print("...done!")
        print(" ") 

        # Analyze test data:
        print("Analyze the test data...")
        
        analyzer.run(test_data)
       
        print("...done! Have a wonderful day!")
        print(" ") 
    #*****************************************


# Run this file via: python utest_residual_analyzer.py
if __name__ == "__main__":
    unittest.main()