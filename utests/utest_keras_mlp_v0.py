import shutil
import unittest
import numpy as np
from jlab_datascience_toolkit.models import make as make_model


class TestKerasMLP(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.configs = {
            "registered_name": "KerasMLP_v0",
            "input_dim": 4,
            "layers_dicts": [
                {
                    "layer_type": "Dense",
                    "layer_configs": {"units": 10, "activation": "relu"},
                },
                {"layer_type": "BatchNormalization"},
                {"layer_type": "Dropout", "layer_configs": {"rate": 0.05}},
                {
                    "layer_type": "Dense",
                    "layer_configs": {"units": 3, "activation": "softmax"},
                },
            ],
        }
        cls.model = make_model(cls.configs["registered_name"], configs=cls.configs)
        cls.x = np.random.rand(100, 4)
        cls.model_folder = "./model_folder/"

    def test_predict(self):
        y_pred = self.model.predict(self.x)
        self.assertTrue(y_pred.shape == (100, 3))

    def test_save_and_load(self):
        y_pred_old = self.model.predict(self.x)
        self.model.save(self.model_folder)
        model_new = make_model(self.configs["registered_name"], configs=self.configs)
        model_new.load(self.model_folder)
        y_pred_new = model_new.predict(self.x)
        self.assertTrue(np.array_equal(y_pred_old, y_pred_new))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.model_folder)


if __name__ == "__main__":
    unittest.main()
