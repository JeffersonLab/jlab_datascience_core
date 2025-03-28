import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from jlab_datascience_toolkit.utils.keras_layers.GP_layer import GaussianProcessLayer

@pytest.fixture
def random_input():
    return np.random.randn(32, 10).astype(np.float32)  # Batch size 32, 10 features


def test_layer_instantiation():
    try:
        layer = GaussianProcessLayer()
        assert layer is not None, "Failed to instantiate GaussianProcessLayer."
    except Exception as e:
        pytest.fail(f"Layer instantiation failed: {e}")


def test_forward_pass(random_input):
    layer = GaussianProcessLayer()
    outputs = layer(random_input)
    assert len(outputs) == 2, "Output should contain mean and stddev."
    assert outputs[0].shape == (32, 1), f"Unexpected output shape: {outputs[0].shape}"
    assert outputs[1].shape == (32, 1), f"Unexpected variance shape: {outputs[1].shape}"


def test_gradient_computation(random_input):
    inputs = Input(shape=(10,))
    layer = GaussianProcessLayer()
    outputs = layer(inputs)
    model = Model(inputs, outputs[0])
    model.compile(optimizer="adam", loss="mse")
    loss = model.train_on_batch(random_input, np.random.randn(32, 1))
    assert loss is not None, "Gradient computation failed."


def test_prior_reset():
    layer = GaussianProcessLayer()
    layer.build((None, 10))
    initial_prior = layer.prior.numpy()
    layer.reset_prior()
    reset_prior = layer.prior.numpy()
    assert np.allclose(reset_prior, np.zeros_like(initial_prior)), "Prior reset did not work as expected."


def test_noise_scale_update():
    layer = GaussianProcessLayer()
    layer.build((None, 10))
    new_noise_scale = 0.01
    layer.set_noise_scale(new_noise_scale)
    assert layer.noise_scale.numpy() == pytest.approx(new_noise_scale), "Noise scale update failed."


def test_variance_calculation(random_input):
    layer = GaussianProcessLayer()
    _, variance = layer(random_input)
    assert variance.shape == (32, 1), "Variance output shape mismatch."
    assert np.all(variance.numpy() >= 0), "Variance should be non-negative."

