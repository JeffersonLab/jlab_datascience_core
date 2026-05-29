import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers, initializers
from tensorflow.keras.constraints import Constraint

class ClipByValue(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

class GaussianProcessApproximationLayer(layers.Layer):
    def __init__(
        self,
        n_fourier_features=1024,
        length_scale=1.0,
        noise_scale=0.001,
        constant_scale=1.0,
        n_out=1,
        train_length_scale=True,
        train_noise_scale=False,
        train_constant_scale=False,
        trainable=True,
        name=None,
        isotropic=True,
        scale_features=False,
        momentum=0.1,
        do_custom_cov_update=False,
        sigma_scale=1.0,
        dropout_rate=0.2,
        **kwargs
    ):
        super().__init__(trainable=trainable, name=name, **kwargs)
        
        self.n_fourier_features = n_fourier_features
        self.initial_noise_scale = noise_scale
        self.initial_length_scale = length_scale
        self.initial_constant_scale=constant_scale
        self.n_out = n_out
        self.initial_prior = self.initial_noise_scale * tf.eye(self.n_fourier_features)
        self.trainable = trainable
        self.train_length_scale = train_length_scale
        self.train_noise_scale = train_noise_scale
        self.train_constant_scale = train_constant_scale
        self.isotropic = isotropic
        self.scale_features = scale_features
        self.momentum = momentum
        self.do_custom_cov_update = do_custom_cov_update
        self.initial_sigma_scale = sigma_scale
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        if self.isotropic:
            self.length_scale = self.add_weight(
                shape=(),  # Scalar variable
                initializer=tf.constant_initializer(self.initial_length_scale),
                trainable=self.train_length_scale,
                dtype=tf.float32,
                constraint=ClipByValue(1e-6, 1e6),  # Custom constraint
                name='length_scale'
            )
        
        else:
            self.length_scale = self.add_weight(
                shape=(input_shape[-1],),  # One value for each input feature
                initializer=tf.constant_initializer(self.initial_length_scale),
                trainable=self.train_length_scale,
                dtype=tf.float32,
                constraint=ClipByValue(1e-6, 1e6),  # Custom constraint
                name='length_scale'
            )
            
        self.sigma_scale = self.add_weight(
            shape=(),
            initializer=tf.constant_initializer(self.initial_sigma_scale),
            trainable=False,
            dtype=tf.float32,
            constraint=ClipByValue(1e-6, 1e6),
            name='sigma_scale'
        )

        self.prior = tf.Variable(tf.zeros(shape=(self.n_fourier_features, self.n_fourier_features)), 
                                 trainable=False, 
                                 name='prior'
        )
        self._track_trackable(self.prior, name="prior")

        self.eigvecs = tf.Variable(tf.eye(self.n_fourier_features),
                                   trainable=False,
                                   name='V'
        )
        
        self.eigvals = tf.Variable(tf.ones(self.n_fourier_features),
                                   trainable=False,
                                   name='Lambda'
        )

        self.noise_scale = self.add_weight(shape=(),  # Scalar variable
                                           initializer=tf.constant_initializer(self.initial_noise_scale),
                                           trainable=self.train_noise_scale,
                                           dtype=tf.float32,
                                           constraint=ClipByValue(1e-6, 1e6),
                                           name="noise_scale"
        )
        
        self.rff_map = layers.Dense(self.n_fourier_features // 2,
                                    trainable=False,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1.0),
                                    bias_initializer=initializers.RandomUniform(0, 2 * np.pi),
                                    name='rff_map'
        )
        
        self.rff_output = layers.Dense(self.n_out, use_bias=False, name='GP_mean_pred')
        
        
    def buld(self, input_shape):
        super().build(input_shape)  

    def call(self, inputs, training=False, return_features=False):
        
        x = tf.cast(inputs, tf.float32)
        x = self.length_scale * x
        x = self.rff_map(x)
        x1 = tf.math.cos(x)
        x2 = tf.math.sin(x)
        ffs = tf.concat([x1, x2], axis=-1)

        if self.scale_features:
            ffs = tf.math.sqrt(2.0 / self.n_fourier_features) * ffs

        # ffs = tf.math.sqrt(self.constant_scale) * ffs
        x = self.dropout(ffs)
        output = self.rff_output(x)
        
        new_prior = tf.stop_gradient(ffs)

        batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)

        if training:
            update_prior_op = (
                self.momentum * self.prior + (1 - self.momentum) * (tf.transpose(new_prior) @ new_prior / batch_size)
            )
            self.prior.assign(update_prior_op)
            self.update_cov(self.prior)
        
        variances = self.calc_variance(ffs)

        stddevs = tf.math.sqrt(variances)
        stddevs = tf.math.scalar_mul(self.sigma_scale, stddevs[:, None])
        result = [output, stddevs]
        if return_features:
            result.append(ffs)

        return result

    
    def update_cov(self, prior):
        eigvals, eigvecs = tf.linalg.eigh(prior)  #EigenDecomposition
        eigvals = tf.where(eigvals > 0, eigvals, tf.zeros_like(eigvals))

        self.eigvals.assign(eigvals)
        self.eigvecs.assign(eigvecs)

    def calc_variance(self, ffs):
        P = ffs @ self.eigvecs
        invE = 1 / (self.eigvals + self.noise_scale + 1e-7)

        variances = tf.linalg.diag_part((P * invE) @ tf.transpose(P))
        return self.noise_scale * variances + self.noise_scale

    def reset_prior(self):
        self.prior.assign(tf.zeros_like(self.prior))

    def set_noise_scale(self, var):
        self.noise_scale.assign(var)
