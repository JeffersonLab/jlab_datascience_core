import numpy as np
import pandas as pd


class SplitDataFrame:
    def __init__(self, configs: dict):
        self.feature_columns = configs["feature_columns"]
        self.target_columns = configs.get("target_columns", None)
        self.rows_fractions = configs.get("rows_fractions", [1.0])
        self.random_state = configs.get("random_state", None)
        assert sum(self.rows_fractions) == 1, 'Fractions must add up to 1 !!!'
    
    @staticmethod
    def split_by_columns(
        df: pd.DataFrame,
        feature_columns: list[str] | str,
        target_columns: list[str] | str
    ) -> list[np.ndarray]:
        x = df.loc[:, feature_columns].to_numpy()
        if target_columns is not None:
            y = df.loc[:, target_columns].to_numpy()
            return [x, y]
        return [x]
    
    @staticmethod
    def split_array(arr: np.ndarray, idxs: np.ndarray, rows_fractions: list[float]) -> list[np.ndarray]:
        subarrays = []
        start = 0
        for i, fraction in enumerate(rows_fractions):
            if i == len(rows_fractions) - 1:
                end = len(idxs)
            else:
                end = start + int(fraction * len(idxs))
            assert end > start, f'Could not split array of shape {arr.shape} with fractions {rows_fractions} !!!'
            sub_idxs = idxs[start : end]
            subarrays.append(arr[sub_idxs])
            start = end
        return subarrays
    
    def run(self, df: pd.DataFrame) -> list[np.ndarray]:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        arrays = self.split_by_columns(df, feature_columns=self.feature_columns, target_columns=self.target_columns)
        idxs = np.random.permutation(len(df.index))
        splitted_arrays = []
        for arr in arrays:
            subarrays = self.split_array(arr, idxs, rows_fractions=self.rows_fractions)
            splitted_arrays.extend(subarrays)
        return splitted_arrays