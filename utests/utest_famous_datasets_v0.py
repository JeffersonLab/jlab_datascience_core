import unittest
import pandas as pd
from jlab_datascience_toolkit.data_parsers.famous_datasets_v0 import FamousDatasetsV0


class TestSplitDataFrame(unittest.TestCase):
    def test_iris(self):
        parser = FamousDatasetsV0(configs={'dataset_name': 'iris'})
        df = parser.load_data()
        self.assertTrue(isinstance(df, pd.DataFrame))

if __name__ == "__main__":
    unittest.main()
