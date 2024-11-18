import yaml
import inspect
import numpy as np
import pandas as pd
from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep


class SplitDataFrame(JDSTDataPrep):
    """
    Splits a given pandas DataFrame by columns (feature_columns & target_columns) and converts them to numpy arrays.
    Each array is then splitted by rows according to the given rows_fractions (which must add up to one).
    """

    def __init__(self, configs: dict):
        self.configs = configs
        self.feature_columns = configs.get(
            "feature_columns", None
        )  # If None, all columns are considered
        self.target_columns = configs.get(
            "target_columns", None
        )  # If None, there will be no target array
        self.rows_fractions = configs.get("rows_fractions", [1.0])
        self.random_state = configs.get("random_state", None)
        assert sum(self.rows_fractions) == 1, "Fractions must add up to 1 !!!"

    @staticmethod
    def split_by_columns(
        df: pd.DataFrame,
        feature_columns: list[str] | str,
        target_columns: list[str] | str,
    ) -> list[np.ndarray]:
        if feature_columns is None:
            x = df.to_numpy()
        else:
            x = df.loc[:, feature_columns].to_numpy()
        if target_columns is not None:
            y = df.loc[:, target_columns].to_numpy()
            return [x, y]
        return [x]

    @staticmethod
    def split_array(
        arr: np.ndarray, idxs: np.ndarray, rows_fractions: list[float]
    ) -> list[np.ndarray]:
        subarrays = []
        start = 0
        for i, fraction in enumerate(rows_fractions):
            if i == len(rows_fractions) - 1:
                end = len(idxs)
            else:
                end = start + int(fraction * len(idxs))
            assert (
                end > start
            ), f"Could not split array of shape {arr.shape} with fractions {rows_fractions} !!!"
            sub_idxs = idxs[start:end]
            subarrays.append(arr[sub_idxs])
            start = end
        return subarrays

    def run(self, df: pd.DataFrame) -> list[np.ndarray]:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        arrays = self.split_by_columns(
            df, feature_columns=self.feature_columns, target_columns=self.target_columns
        )
        idxs = np.random.permutation(len(df.index))
        splitted_arrays = []
        for arr in arrays:
            subarrays = self.split_array(arr, idxs, rows_fractions=self.rows_fractions)
            splitted_arrays.extend(subarrays)
        return splitted_arrays

    def get_info(self):
        """Prints this module's docstring."""
        print(inspect.getdoc(self))

    def save_config(self, path: str):
        assert path.endswith(".yaml")
        with open(path, "w") as file:
            yaml.safe_dump(self.configs, file)

    @staticmethod
    def load_config(path: str):
        assert path.endswith(".yaml")
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError

    def save_data(self):
        raise NotImplementedError
