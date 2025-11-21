# ckhronos.py
"""Convolutional Encoder + KHRONOS Head"""

import jax
import jax.numpy as jnp
from flax import linen as nn

def init_weights_prod(key, shape, dtype=jnp.float32):
    return (jax.random.normal(key, shape, dtype) * 0.1) + 1.0


class KHRONOSLayer(nn.Module):
    kdims: int
    kelem: int
    krank: int

    def setup(self):
        self.grid = jnp.linspace(0, 1, self.kelem)[None, None, :]
        self.weights = self.param("weights", init_weights_prod, (self.kdims, self.kelem, self.krank))
        self.scale = self.param("scale", lambda k, s: jnp.ones(s) * (self.kelem - 1), (1, self.kdims, 1))

    @nn.compact
    def __call__(self, x):
        x_expanded = x[..., None]
        d = jnp.abs(x_expanded - self.grid) * self.scale

        basis_activations = jnp.where(
            d < 0.5,
            0.75 - d**2,
            jnp.where((d >= 0.5) & (d < 1.5), 0.5 * (1.5 - d) ** 2, 0.0),
        )

        per_dim_output = jnp.einsum("bdg, dgm -> bdm", basis_activations, self.weights)

        per_dim_output_safe = jnp.abs(per_dim_output) + 1e-9
        sum_log = jnp.sum(jnp.log(per_dim_output_safe), axis=1)
        is_negative = (per_dim_output < 0).astype(jnp.int32)
        num_negative = jnp.sum(is_negative, axis=1)
        prod_sign = 1.0 - 2.0 * (num_negative % 2)

        return prod_sign * jnp.exp(sum_log)


class Encoder(nn.Module):
    """(28, 28, 1) -> (latent_dim,)"""

    latent_dim: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.latent_dim)(x)
        return x


class KHRONOSClassifierHead(nn.Module):
    """(latent_dim,) -> (num_classes,)"""

    kelem: int
    krank: int
    kouts: int
    kdims: int
    compute_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        kfeatures_norm = nn.sigmoid(x.astype(self.compute_dtype))
        basis = KHRONOSLayer(kdims=self.kdims, kelem=self.kelem, krank=self.krank)(kfeatures_norm)
        out = nn.Dense(features=self.kouts, name="head", dtype=self.compute_dtype)(basis)
        if self.kouts == 1:
            return jnp.squeeze(out, axis=-1)
        return out


class KHRONOSRegressorHead(nn.Module):
    """(latent_dim,) -> regression."""

    kelem: int
    krank: int
    kdims: int
    compute_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        kfeatures_norm = nn.sigmoid(x.astype(self.compute_dtype))
        basis = KHRONOSLayer(kdims=self.kdims, kelem=self.kelem, krank=self.krank)(kfeatures_norm)
        out = nn.Dense(features=1, name="head_reg", dtype=self.compute_dtype)(basis)
        return jnp.squeeze(out, axis=-1)


class ConvKHRONOS(nn.Module):

    kdims: int
    kelem: int
    krank: int
    kouts: int
    compute_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self._encoder = Encoder(latent_dim=self.kdims)
        self._head = KHRONOSClassifierHead(
            kelem=self.kelem, krank=self.krank, kouts=self.kouts, kdims=self.kdims, compute_dtype=self.compute_dtype
        )

    def __call__(self, x, train: bool = True):
        kfeatures = self._encoder(x.astype(self.compute_dtype), train=train)
        logits = self._head(kfeatures, train=train)
        return logits

    def encoder(self, x, train: bool = True):
        return self._encoder(x, train=train)

    def head(self, z, train: bool = True):
        return self._head(z, train=train)


class ConvKHRONOSRegressor(nn.Module):

    kdims: int
    kelem: int
    krank: int
    compute_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self._encoder = Encoder(latent_dim=self.kdims)
        self._head = KHRONOSRegressorHead(
            kelem=self.kelem, krank=self.krank, kdims=self.kdims, compute_dtype=self.compute_dtype
        )

    def __call__(self, x, train: bool = True):
        kfeatures = self._encoder(x.astype(self.compute_dtype), train=train)
        out = self._head(kfeatures, train=train)
        return out

    def encoder(self, x, train: bool = True):
        return self._encoder(x, train=train)

    def head(self, z, train: bool = True):
        return self._head(z, train=train)


def count_parameters(params_tree) -> int:
    return sum(p.size for p in jax.tree_util.tree_leaves(params_tree))
