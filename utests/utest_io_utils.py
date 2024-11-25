from jlab_datascience_toolkit.utils.io import save_yaml_config, load_yaml_config
from pathlib import Path
import unittest
import logging
import tempfile
import random
import string
import shutil
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def generate_random_string(length):
    alphanumeric = string.ascii_letters + string.digits
    return "".join(random.choice(alphanumeric) for _ in range(length))


class TestIOUtils(unittest.TestCase):

    # Initialize:
    # *****************************************
    def __init__(self, *args, **kwargs):
        super(TestIOUtils, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(self) -> None:
        print("Setting up all tests...")
        self.config = {"name": "test", "scale": 1, "list_example": [0.1, 1.2, 2.3]}
        self.test_path = Path("./temp_dir_" + generate_random_string(6))
        self.existing_file = self.test_path.joinpath("existing_file.yaml")

    @classmethod
    def tearDownClass(self) -> None:
        print("\nHave a good day!")

    def setUp(self) -> None:
        print(
            "\n----------------------------------------------------------------------"
        )
        os.makedirs(self.test_path)
        with open(self.existing_file, "w"):
            pass
        return super().setUp()

    def tearDown(self) -> None:
        # print('\nEnd of Test')
        print("----------------------------------------------------------------------")
        shutil.rmtree(self.test_path)
        return super().tearDown()

    def test_save_load_with_dir(self):
        save_yaml_config(self.config, self.test_path)
        self.assertTrue(self.test_path.joinpath("config.yaml").exists())
        config = load_yaml_config(self.test_path)
        for k in self.config:
            self.assertEqual(self.config[k], config[k])

    def test_save_existing_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            save_yaml_config(self.config, self.existing_file)

    def test_load_not_existing(self):
        with self.assertRaises(FileNotFoundError):
            load_yaml_config(self.test_path.joinpath("no_file_exists_here.yaml"))

    def test_save_load_filename(self):
        new_filename = self.test_path.joinpath("new_config.yaml")
        save_yaml_config(self.config, new_filename)
        load_yaml_config(new_filename)

    def test_overwrite_filename(self):
        # We will simply try saving the same thing three times. First should succeed,
        # second should fail with overwrite==False, third should succeed with overwrite==True
        new_filename = self.test_path.joinpath("new_config.yaml")
        save_yaml_config(self.config, new_filename)
        with self.assertRaises(FileExistsError):
            save_yaml_config(self.config, new_filename)
        config = self.config.copy()
        config["name"] = "train"
        save_yaml_config(config, new_filename, overwrite=True)
        loaded_config = load_yaml_config(new_filename)
        for k in config:
            self.assertEqual(config[k], loaded_config[k])


# Run this file via: python utest_io_utils.py
if __name__ == "__main__":
    unittest.main()
