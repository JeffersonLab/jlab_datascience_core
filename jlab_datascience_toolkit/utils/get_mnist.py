from jlab_datascience_toolkit.utils.check_internet_connection import internet_available

import tensorflow as tf
import numpy as np
import os

def get_mnist_data():
    """
    Load MNIST data based on internet availability.
    """
    if internet_available():
        try:
            (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
            print("MNIST dataset has been downloaded.")
        except Exception as e:
            print(f"Failed to download MNIST dataset due to an error: {e}")
            x_train, y_train, x_val, y_val = load_local_mnist()
    else:
        print("Trying to load dataset locally...")
        x_train, y_train, x_val, y_val = load_local_mnist()

    return x_train, y_train, x_val, y_val

def load_local_mnist():
    """
    Helper function to load MNIST data locally; assumes data is within the repo.
    """
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_file_path, '..', 'data', 'example_data', 'mnist_data')

    if os.path.exists(data_path):
        x_train = np.load(os.path.join(data_path, 'x_train.npy'))
        x_val = np.load(os.path.join(data_path, 'x_val.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        y_val = np.load(os.path.join(data_path, 'y_val.npy'))
        print("... Loaded MNIST data from local file.")
    else:
        raise FileNotFoundError("Local MNIST file not found. Please check the path.")
    return x_train, y_train, x_val, y_val