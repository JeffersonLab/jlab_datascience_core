import unittest
import numpy as np
import pandas as pd
from jlab_datascience_toolkit.data_prep.split_dataframe_v0 import SplitDataFrame


class TestSplitDataFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a = np.arange(start=10, stop=20, step=1)
        cls.b = np.arange(start=20, stop=30, step=1)
        cls.c = np.arange(start=30, stop=40, step=1)
        cls.d = np.arange(start=40, stop=50, step=1)
        cls.df = pd.DataFrame({"a": cls.a, "b": cls.b, "c": cls.c, "d": cls.d})
        cls.feature_columns = ["a", "b"]
        cls.target_columns = ["c", "d"]
        return super().setUpClass()

    def test_split_by_columns(self):
        arrays = SplitDataFrame.split_by_columns(
            df=self.df,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
        )
        self.assertTrue(len(arrays) == 2)
        self.assertTrue(np.array_equal(arrays[0], np.stack([self.a, self.b], axis=-1)))
        self.assertTrue(np.array_equal(arrays[1], np.stack([self.c, self.d], axis=-1)))

    def test_split_array(self):
        arr = np.stack([self.a, self.b], axis=-1)
        idxs = np.array([0, 4, 3, 8, 1, 5, 2, 7, 6, 9])
        rows_fractions = [0.6, 0.2, 0.2]
        arrays = SplitDataFrame.split_array(
            arr=arr, idxs=idxs, rows_fractions=rows_fractions
        )
        self.assertTrue(len(arrays) == len(rows_fractions))
        self.assertTrue(np.array_equal(arrays[0], arr[idxs[:6], :]))
        self.assertTrue(np.array_equal(arrays[1], arr[idxs[6:8], :]))
        self.assertTrue(np.array_equal(arrays[2], arr[idxs[8:], :]))


if __name__ == "__main__":
    unittest.main()
