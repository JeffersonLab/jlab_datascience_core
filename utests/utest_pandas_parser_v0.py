from jlab_datascience_toolkit.data_parser import make
import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

rng = np.random.default_rng(seed=42)


class TestPandasParserv0(unittest.TestCase):

    # Initialize:
    # *****************************************
    def __init__(self, *args, **kwargs):
        super(TestPandasParserv0, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(self) -> None:
        print('Setting up all tests...')
        self.columns = ['R121GMES', 'R122GMES',
                        'R123GMES', 'R121GSET', 'R122GSET', 'R123GSET']
        self.path = './pandas_parser_utest.csv'
        self.samples = 100
        data = rng.normal(loc=5, scale=1, size=(
            self.samples, len(self.columns)))
        dates = []
        for i in range(1, self.samples+1):
            dates.append(np.datetime64(
                f'2010-03-24 10:{i//60:02d}:{i % 60:02d}'))

        test_data = pd.DataFrame(data, columns=self.columns, index=dates)
        test_data.index.name = 'Date'
        test_data
        test_data.to_csv(self.path)

    @classmethod
    def tearDownClass(self) -> None:
        print('Removing temporary files...')
        os.remove(self.path)
        print('Have a good day!')

    def setUp(self) -> None:
        print()
        return super().setUp()

    def tearDown(self) -> None:
        print()
        return super().tearDown()

    def test_no_config(self):
        print('*****No Config Test*****\n')
        parser = make('PandasParser_v0')
        output = parser.load_data()
        self.assertIsNone(output)

    def test_string_filepaths(self):
        print('*****String Filepaths Test*****\n')

        parser = make('PandasParser_v0', config=dict(
            filepaths='./pandas_parser_utest.csv'))
        output = parser.load_data()
        print('Output Head:\n', output.head())

        self.assertEqual(output.shape, (self.samples, len(self.columns)+1))

    def test_one_item_list_filepaths(self):
        print('*****One Item List Test*****\n')

        parser = make('PandasParser_v0', config=dict(
            filepaths=['./pandas_parser_utest.csv']))
        output = parser.load_data()
        print('Output Head:\n', output.head())
        self.assertEqual(output.shape, (self.samples, len(self.columns)+1))

    def test_usecols_read_arg(self):
        print('*****Usecols Read Arg Test*****\n')

        two_columns = ['R121GMES', 'R121GSET']
        parser = make('PandasParser_v0', config=dict(
            filepaths='./pandas_parser_utest.csv', read_kwargs=dict(usecols=two_columns)))
        output = parser.load_data()
        print('Output Head:\n', output.head())
        self.assertEqual(output.shape, (self.samples, 2))
        self.assertEqual(set(output.columns), set(two_columns))

    def test_use_datetime_index(self):
        print('*****Use Datetime Index Test*****\n')

        def column_lambda(x): return ('GMES' in x) or (x == 'Date')
        read_kwargs = dict(usecols=column_lambda,
                           index_col='Date', parse_dates=True)
        parser = make('PandasParser_v0',
                      config=dict(
                          filepaths='./pandas_parser_utest.csv', read_kwargs=read_kwargs)
                      )
        output = parser.load_data()
        print('Output Head:\n', output.head())
        self.assertEqual(output.shape, (self.samples, 3))
        for column in output.columns:
            self.assertTrue('GMES' in column)
        self.assertIsInstance(output.index, pd.DatetimeIndex)


# Run this file via: python utest_pandas_parser_v0.py
if __name__ == "__main__":
    unittest.main()
