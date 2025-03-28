import pytest
import numpy as np
import tensorflow as tf
from jlab_datascience_toolkit.utils.keras_losses.nll_loss import gp_nll_loss

@pytest.fixture
def random_data():
    """random input data for testing."""
    pred = tf.random.normal((32, 1))  # Predicted mean
    y = tf.random.normal((32, 1))     # Ground truth
    std = tf.random.uniform((32, 1), minval=0.1, maxval=1.0)  # Standard deviation
    noise_scale = 0.01
    return pred, y, std, noise_scale

def test_basic_loss_computation(random_data):
    """basic loss computation without errors."""
    pred, y, std, noise_scale = random_data
    loss = gp_nll_loss(pred, y, std, noise_scale)
    assert loss is not None, "Loss computation returned None"
    assert tf.is_tensor(loss), "Loss computation did not return a tensor"
    assert loss >= 0, "Loss should be non-negative"

def test_zero_variance_handling():
    """handling of zero variance."""
    pred = tf.constant([[0.5], [1.0], [-0.5]], dtype=tf.float32)
    y = tf.constant([[0.5], [1.0], [-0.5]], dtype=tf.float32)
    std = tf.constant([[0.0], [0.0], [0.0]], dtype=tf.float32)  # Zero variance
    noise_scale = 0.01
    loss = gp_nll_loss(pred, y, std, noise_scale)
    assert loss is not None, "Loss computation returned None"
    assert tf.is_tensor(loss), "Loss computation did not return a tensor"