import jlab_datascience_toolkit.data_parsers as parsers
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os

class UTestNumpyParser(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestNumpyParser,self).__init__(*args, **kwargs)

        # Get an into:
        print(" ")
        print("*******************************")
        print("*                             *")
        print("*   Unit Test: Numpy Parser   *")
        print("*                             *")
        print("*******************************")
        print(" ")
    #*****************************************
    
    # Test drive the parser:
    #*****************************************
    def test_drive_numpy_parser(self):
        print("Create test data set(s)...")

        # We need to create test data set(s) first which we will then try to load:
        n_sets = 3
        n_events = 5000
        # Start with the names that we will use later for the numpy parser:
        data_locs = ['data_'+str(i)+'.npy' for i in range(n_sets)]
        # Features of the data, that are not important for this test:
        data_means = [-5.0,0.0,5.0]
        data_widths = [0.5]*n_sets

        # Create and store the test data set(s)
        #+++++++++++++++++++
        for i in range(n_sets):
            np.save(data_locs[i],np.random.normal(data_means[i],data_widths[i],size=n_events))
        #+++++++++++++++++++

        print("...done!")
        print(" ")

        # Now we load our data parser. The default config does not have the data locations
        # so we need to provide an additional config that allows us to overwrite the default setting (which is simply "")
        print("Load numpy parser...")

        parser_cfg = {'data_loc':data_locs}
        this_file_loc = os.path.dirname(__file__)
        cfg_loc = os.path.join(this_file_loc,'../jlab_datascience_toolkit/cfgs/defaults/numpy_parser_cfg.yaml')
        npy_parser = parsers.make("NumpyParser_v0",path_to_cfg=cfg_loc,user_config=parser_cfg)

        # Lets see if we can call the information about this module:
        npy_parser.get_info()

        print("...done!")
        print(" ")

        # Load the data:
        print("Load the test data set(s)...")

        test_data = npy_parser.load_data()

        print("...done!")
        print(" ")

        # Run a consistency check:
        # We have three data sets with 5k events each. Thus, we expect the total data set to have the shape: 15000
        print("Run dimensional check on parsed data...")

        passDimensionCheck = False
        if test_data.shape[0] == n_events*n_sets:
            passDimensionCheck = True 

        print("...done!")
        print(" ")

        # Plot the test data:
        print("Visualize data...")

        plt.rcParams.update({'font.size':20})
        fig, ax = plt.subplots(figsize=(12,8))

        ax.hist(test_data,100)
        ax.set_xlabel('Data')
        ax.set_ylabel('Entries')
        ax.grid(True)

        fig.savefig('numpy_parser_data.png')
        plt.close(fig)

        print("...done!")
        print(" ")

        # Clean up everything:
        print("Remove test data set(s)...")

        #+++++++++++++++++++
        for i in range(n_sets):
          os.remove(data_locs[i])
        #+++++++++++++++++++

        print("...done!")
        print(" ")
        
        # Check that we passed the dimension test:
        self.assertTrue(passDimensionCheck)

        print("Have a great day!")
    #*****************************************

# Run this file via: python utest_numpy_parser.py
if __name__ == "__main__":
    unittest.main()
