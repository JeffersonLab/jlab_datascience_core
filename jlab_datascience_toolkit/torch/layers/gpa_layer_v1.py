"""
PyTorch port of GaussianProcessApproximationLayer (Random Fourier Features GP approx).

Key differences from the Keras version, all forced by PyTorch's eager parameter
construction (no lazy `build(input_shape)`):
  - `input_dim` is now a required constructor argument. The original isotropic=False
    branch referenced `input_shape` inside `__init__`, where it was never actually in
    scope (would have raised NameError if that branch were exercised). `rff_map` also
    needs `in_features` up front here, so input_dim is required unconditionally.
  - Keras' `constraint=ClipByValue(...)` reprojects a weight after every optimizer
    step. PyTorch has no built-in analog, so this port exposes `clip_parameters()` —
    call it right after `optimizer.step()` (same pattern as WGAN critic clipping).
  - `training` follows standard PyTorch convention: if not passed explicitly, it
    falls back to `self.training` (set via `.train()` / `.eval()`), so `nn.Dropout`
    and the prior-update logic stay in sync automatically.
"""

import math
import torch
import torch.nn as nn


class GaussianProcessApproximationLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_fourier_features: int = 1024,
        length_scale: float = 1.0,
        noise_scale: float = 0.001,
        constant_scale: float = 1.0,
        n_out: int = 1,
        train_length_scale: bool = True,
        train_noise_scale: bool = False,
        train_constant_scale: bool = False,
        isotropic: bool = True,
        scale_features: bool = False,
        momentum: float = 0.5,
        sigma_scale: float = 0.01,
        dropout_rate: float = 0.2,
        activation: any = None
    ):
        super().__init__()

        assert n_fourier_features % 2 == 0, "n_fourier_features must be even (split across cos/sin)."

        self.input_dim = input_dim
        self.n_fourier_features = n_fourier_features
        self.n_out = n_out
        self.isotropic = isotropic
        self.scale_features = scale_features
        self.momentum = momentum
        self.train_length_scale = train_length_scale
        self.train_noise_scale = train_noise_scale

        # constant_scale / train_constant_scale are stored for parity with the
        # original but were never actually consumed there either (the line that
        # would use it is commented out in the source). Kept as plain attributes,
        # not a weight, matching original behavior.
        self.constant_scale = constant_scale
        self.train_constant_scale = train_constant_scale

        # ---- length_scale: trainable by default ----
        ls_shape = () if isotropic else (input_dim,)
        ls_init = torch.full(ls_shape, float(length_scale))
        if train_length_scale:
            self.length_scale = nn.Parameter(ls_init)
        else:
            self.register_buffer("length_scale", ls_init)

        # ---- sigma_scale: always non-trainable in the original (hardcoded
        # trainable=False regardless of train_constant_scale) ----
        self.register_buffer("sigma_scale", torch.tensor(float(sigma_scale)))

        # ---- noise_scale: non-trainable by default ----
        ns_init = torch.tensor(float(noise_scale))
        if train_noise_scale:
            self.noise_scale = nn.Parameter(ns_init)
        else:
            self.register_buffer("noise_scale", ns_init)

        # ---- running prior + its eigendecomposition: state, not weights ----
        self.register_buffer("prior", torch.zeros(n_fourier_features, n_fourier_features))
        self.register_buffer("eigvecs", torch.eye(n_fourier_features))
        self.register_buffer("eigvals", torch.ones(n_fourier_features))
        # unused elsewhere in the original (dead code kept only for parity)
        self.register_buffer("initial_prior", noise_scale * torch.eye(n_fourier_features))

        # ---- frozen random projection (the "RF" in RFF) ----
        self.rff_map = nn.Linear(input_dim, n_fourier_features // 2)
        with torch.no_grad():
            self.rff_map.weight.normal_(mean=0.0, std=1.0)
            self.rff_map.bias.uniform_(0.0, 2 * math.pi)
        for p in self.rff_map.parameters():
            p.requires_grad_(False)

        # ---- trainable linear GP-mean readout ----
        self.rff_output = nn.Linear(n_fourier_features, n_out, bias=False)
        nn.init.xavier_uniform_(self.rff_output.weight)

        self.dropout = nn.Dropout(dropout_rate)
        self.output_activation = activation

    @torch.no_grad()
    def clip_parameters(self, min_value: float = 1e-6, max_value: float = 1e6):
        """
        Equivalent of the Keras ClipByValue constraint. Call this after every
        optimizer.step() for any of length_scale / noise_scale you're training,
        e.g.:
            loss.backward()
            optimizer.step()
            gp_layer.clip_parameters()
        """
        if self.train_length_scale:
            self.length_scale.clamp_(min_value, max_value)
        if self.train_noise_scale:
            self.noise_scale.clamp_(min_value, max_value)
        self.sigma_scale.clamp_(min_value, max_value)

    def forward(self, inputs, training: bool = None, return_features: bool = False):
        if training is None:
            training = self.training

        x = inputs.float()
        x = self.length_scale * x
        x = self.rff_map(x)
        x1 = torch.cos(x)
        x2 = torch.sin(x)
        ffs = torch.cat([x1, x2], dim=-1)

        if self.scale_features:
            ffs = math.sqrt(2.0 / self.n_fourier_features) * ffs

        x = self.dropout(ffs)
        output = self.rff_output(x)
        if self.output_activation is not None:
            output = self.output_activation(output)

        new_prior = ffs.detach()  # stop_gradient
        batch_size = float(inputs.shape[0])

        if training:
            with torch.no_grad():
                updated_prior = self.momentum * self.prior + (1 - self.momentum) * (
                    new_prior.T @ new_prior / batch_size
                )
                self.prior.copy_(updated_prior)
            self.update_cov(self.prior)

        variances = self.calc_variance(ffs)
        stddevs = torch.sqrt(variances)
        stddevs = self.sigma_scale * stddevs.unsqueeze(-1)

        result = [output, stddevs]
        if return_features:
            result.append(ffs)
        return result

    @torch.no_grad()
    def update_cov(self, prior):
        eigvals, eigvecs = torch.linalg.eigh(prior)  # ascending order, same as tf.linalg.eigh
        eigvals = torch.clamp(eigvals, min=0.0)
        self.eigvals.copy_(eigvals)
        self.eigvecs.copy_(eigvecs)

    def calc_variance(self, ffs):
        P = ffs @ self.eigvecs
        invE = 1.0 / (self.eigvals + self.noise_scale + 1e-7)
        # row-wise dot product == diag_part((P * invE) @ P.T), avoids materializing
        # the full (batch, batch) matrix that the TF version implicitly built.
        variances = torch.sum((P * invE) * P, dim=-1)
        return self.noise_scale * variances + self.noise_scale

    @torch.no_grad()
    def reset_prior(self):
        self.prior.zero_()

    @torch.no_grad()
    def set_noise_scale(self, var):
        self.noise_scale.copy_(torch.as_tensor(var, dtype=self.noise_scale.dtype))