import tensorflow as tf
import numpy as np

def gp_nll_loss(pred, y, std, noise_scale):
    """
    Negative Log-Likelihood (NLL) Loss for Gaussian Process (GP) regression.

    Args:
        pred (tf.Tensor): Predicted mean of the GP model (shape: [batch_size, 1]).
        y (tf.Tensor): Ground truth target values (shape: [batch_size, 1]).
        std (tf.Tensor): Standard deviation predictions from the GP model (shape: [batch_size, 1]).
        noise_scale (tf.Tensor or float): Noise scale (variance) of the GP model.

    Returns:
        tf.Tensor: The computed NLL loss (scalar).
    """
    
    sigma_star = tf.square(std) + noise_scale + 1e-5  # Adding a small constant for numerical stability

    # NLL calculation
    term1 = tf.math.log(2 * np.pi * sigma_star)
    term2 = tf.square(pred - y) / sigma_star
    loss = tf.reduce_mean(0.5 * (term1 + term2))

    return loss

