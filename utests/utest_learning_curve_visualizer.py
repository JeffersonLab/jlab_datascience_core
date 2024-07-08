import jlab_datascience_toolkit.analysis as analyses
import numpy as np
import unittest
import matplotlib.pyplot as plt
import os

class UTestLearningCurveVisualizer(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestLearningCurveVisualizer,self).__init__(*args, **kwargs)
        
        # Get an intro:
        print(" ")
        print("*******************************************")
        print("*                                         *")
        print("*   Learning Curve Visualizer Unit-Test   *")
        print("*                                         *")
        print("*******************************************")
        print(" ")
    #*****************************************
    
    # Test everything:
    #*****************************************
    def test_learning_curve_visualizer(self):
    

        # Create mock data:
        print("Create test data...")
        x = np.linspace(0,11,11) 

        test_data = {
            'data_1':0.9*x,
            'data_2':1.1*x,
            'data_3':x*x,
            'data_4':0.5*x*x
        }

        print("...done!")
        print(" ")
         

        # Load the analyzer:
        print("Load analyzer module...")

        # We need the name:
        module_id = "LearningCurveVisualizer_v0"
        
        # Get the path to the default configuration:
        this_file_loc = os.path.dirname(__file__)
        cfg_loc = os.path.join(this_file_loc,'../jlab_datascience_toolkit/cfgs/defaults/learning_curve_visualizer_cfg.yaml')

        # Specify the name of the output file:
        user_cfg = {
            'output_loc': 'results_utest_learning_curve_visualizer',
            'plots':{'plot_a':['data_1','data_2'],'plot_b':['data_3','data_4']},
            'plot_labels':{'plot_a':['Trial','Some Value'],'plot_b':['Trial','Different Values']},
            'plot_legends':{'plot_a':['Data 1','Data 2'],'plot_b':['Data 3','Data 4']},
            'plot_names':{'plot_a':'first_plot','plot_b':'second_plot'}
        }
        
        analyzer = analyses.make(module_id,path_to_cfg=cfg_loc,user_config=user_cfg)

        print("...done!")
        print(" ") 

        print("Visualize test data...")

        analyzer.run(test_data)

        print("..done! Have a wonderful day!")
        print(" ")

       
    #*****************************************


# Run this file via: python utest_learning_curve_visualizer.py
if __name__ == "__main__":
    unittest.main()