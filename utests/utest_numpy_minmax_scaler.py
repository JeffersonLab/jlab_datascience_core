import jlab_datascience_toolkit.data_prep as preps
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


class UTestNumpyMinMaxScaler(unittest.TestCase):

    # Initialize:
    # *****************************************
    def __init__(self, *args, **kwargs):
        super(UTestNumpyMinMaxScaler, self).__init__(*args, **kwargs)

        # Get an into:
        print(" ")
        print("***************************************")
        print("*                                     *")
        print("*   Unit Test: Numpy Min Max Scaler   *")
        print("*                                     *")
        print("***************************************")
        print(" ")

    # *****************************************

    # Test the min max scaler:
    # *****************************************
    def test_drive_numpy_minmax_scaler(self):
        # Create some data first, that we wish to scale:
        print("Create test data...")

        test_data = np.random.uniform(5.0, 10.0, size=(5000, 1))

        print("...done!")
        print("  ")

        # Now load the scaler by defining a user config first:
        print("Load numpy min max scaler...")

        this_file_loc = os.path.dirname(__file__)
        cfg_loc = os.path.join(
            this_file_loc,
            "../jlab_datascience_toolkit/cfgs/defaults/numpy_minmax_scaler_cfg.yaml",
        )
        param_store_loc = this_file_loc + "/numpy_minmax_scaler_params"
        scaler_cfg = {"feature_range": (-1.0, 1.0), "store_loc": param_store_loc}
        npy_scaler = preps.make(
            "NumpyMinMaxScaler_v0", path_to_cfg=cfg_loc, user_config=scaler_cfg
        )

        # Print the module info:
        npy_scaler.get_info()

        print("...done!")
        print("  ")

        # Run the scaler:
        print("Scale data...")

        scaled_data = npy_scaler.run(test_data)

        print("...done!")
        print("  ")

        # Undo the scaling:
        print("Reverse scaling...")

        unscaled_data = npy_scaler.reverse(scaled_data)

        print("...done!")
        print("  ")

        # Check if the data ranges make sense at all:
        pass_range_check_1 = False
        pass_range_check_2 = False

        print("Run sanity checks...")

        # Check scaled data:
        if round(np.min(scaled_data), 1) == -1.0 and round(np.max(scaled_data)) == 1.0:
            pass_range_check_1 = True

        # Check if the unscaled data has the same limits as the original test data:
        if round(np.min(test_data), 1) == round(np.min(unscaled_data), 1) and round(
            np.max(test_data), 1
        ) == round(np.max(unscaled_data), 1):
            pass_range_check_2 = True

        print("...done!")
        print("  ")

        # Store and load the scaler parameters --> We want to see that the module checkpointing is working
        print("Store and retreive scaler parameters...")

        pass_checkpointing = False
        # Store the params:
        npy_scaler.save()

        # And read them back in:
        param_dict = npy_scaler.load()

        # If everything went right, there should be a file with scaling parameters and the param dictionary
        # should not be empty:
        if os.path.exists(scaler_cfg["store_loc"]) and bool(param_dict):
            pass_checkpointing = True

        print("...done!")
        print("  ")

        # Clean up:
        print("Remove created data...")

        shutil.rmtree("numpy_minmax_scaler_params")

        print("...done!")
        print("  ")

        # Test if the type checker is working:
        passTypeChecker = False
        print("Test type checker (an error message should show up below this line)...")

        val = npy_scaler.run([1, 2, 3, 4])

        if val is None:
            passTypeChecker = True

        print("...done!")
        print("  ")

        self.assertTrue(
            pass_range_check_1
            & pass_range_check_2
            & pass_checkpointing
            & passTypeChecker
        )

        print("Have a great day!")

    # *****************************************


# Run this file via: python utest_numpy_parser.py
if __name__ == "__main__":
    unittest.main()
