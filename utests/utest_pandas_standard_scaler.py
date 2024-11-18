from jlab_datascience_toolkit.data_prep import make
import unittest
import logging
import matplotlib.pyplot as plt
import inspect
import pandas as pd
import numpy as np
import shutil
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
rng = np.random.default_rng(seed=42)
prep_id = "PandasStandardScaler_v0"


class TestPandasStandardScalerv0(unittest.TestCase):

    # Initialize:
    # *****************************************
    def __init__(self, *args, **kwargs):
        super(TestPandasStandardScalerv0, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(self) -> None:
        print("Setting up all tests...")

    @classmethod
    def tearDownClass(self) -> None:
        print("\nHave a good day!")

    def setUp(self) -> None:
        x1 = pd.Series(rng.normal(loc=1, scale=2, size=(100,)), name="X1")
        x2 = pd.Series(rng.normal(loc=3, scale=4, size=(100,)), name="X2")
        x3 = pd.Series(rng.uniform(low=3, high=10, size=(100,)), name="X3")
        x4 = pd.Series(rng.uniform(low=-4, high=1, size=(100,)), name="X4")
        data = pd.concat([x1, x2, x3, x4], axis=1)
        self.data = data
        print(
            "\n----------------------------------------------------------------------"
        )
        return super().setUp()

    def tearDown(self) -> None:
        # print('\nEnd of Test')
        print("----------------------------------------------------------------------")
        return super().tearDown()

    def test_output_types(self):

        prep = make(prep_id, config={"inplace": True})
        output = prep.run(self.data)
        self.assertIsNone(output)

        prep = make(prep_id, config={"inplace": False})
        output = prep.run(self.data)
        self.assertEqual(
            type(output), pd.DataFrame, msg="Output not DataFrame when inplace==False"
        )

    def test_axis_zero(self):
        prep = make(prep_id, config={"axis": 0})
        scaled_data = prep.run(self.data)
        mean = scaled_data.mean(axis=0)
        var = scaled_data.var(axis=0, ddof=0)
        self.assertTrue(
            np.allclose(mean, np.zeros_like(mean)), msg="Column mean not equal to zero"
        )
        self.assertTrue(
            np.allclose(var, np.ones_like(var)), msg="Column variance not equal to one"
        )

    def test_inplace_run(self):
        prep = make(prep_id, config={"inplace": True})
        out = prep.run(self.data)
        mean = self.data.mean(axis=0)
        var = self.data.var(axis=0, ddof=0)
        self.assertTrue(
            np.allclose(mean, np.zeros_like(mean)), msg="Column mean not equal to zero"
        )
        self.assertTrue(
            np.allclose(var, np.ones_like(var)), msg="Column variance not equal to one"
        )

    def test_zero_variance(self):
        original_shape = self.data.shape
        self.data["X5"] = pd.Series(4 * np.ones(shape=(100,)), name="X5")
        prep = make(prep_id)
        scaled_data = prep.run(self.data)
        mean = scaled_data.mean(axis=0)
        var = scaled_data.var(axis=0, ddof=0)
        self.assertTrue(
            np.allclose(mean, np.zeros_like(mean)), msg="Column mean not equal to zero"
        )

        theory_var = np.ones_like(var)
        theory_var[-1] = 0
        self.assertTrue(
            np.allclose(var, theory_var), msg="Scaled variance is incorrect"
        )

    def test_multi_run(self):
        # Should set mean and scale only based on first dataset called with run
        prep = make(prep_id)
        scaled_data = prep.run(self.data)
        saved_mean = prep.mean
        saved_scale = prep.scale

        scaled_data2 = prep.run(self.data + 5)
        self.assertTrue(
            (saved_mean == prep.mean).all(), msg="Mean has changed after second run()"
        )
        self.assertTrue(
            (saved_scale == prep.scale).all(),
            msg="Scale has changed after second run()",
        )

        # Mean in data+5 after scaling should be 5 / scale
        self.assertTrue(
            np.allclose(scaled_data2.mean(), 5 / prep.scale),
            msg="Mean of second run() is incorrect.",
        )

    def test_save_load(self):
        prep = make(prep_id)
        scaled_data = prep.run(self.data)
        save_path = "./test_saved_prep"
        try:
            prep.save(save_path)
            new_prep = make(prep_id)
            new_prep.load(save_path)
            new_scaled_data = new_prep.run(self.data)
            self.assertTrue(
                np.allclose(new_scaled_data, scaled_data),
                msg="Scaled data after load() does not match",
            )
        finally:
            shutil.rmtree(save_path)

    def test_reverse_scaling(self):
        prep = make(prep_id)
        scaled_data = prep.run(self.data)
        unscaled_data = prep.reverse(scaled_data)

        self.assertTrue(np.allclose(self.data, unscaled_data))


# Run this file via: python utest_csv_parser_v0.py
if __name__ == "__main__":
    unittest.main()
