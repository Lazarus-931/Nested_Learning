import math
from dataclasses import dataclass

import jax
import optax
import flax
import optax.schedules as sche
import flax.linen as nn
import flax.nnx as nnx
import jax.numpy as jnp
import einops

@dataclass
class HYPER_CMS_config:
    dim: int
    depth: int
    residual_streams: int
    rate: int
    layer_id: int
    dynamic: bool

config = HYPER_CMS_config(
    dim = 4,
    depth = 4,
    residual_streams = 4,
    rate = 2,
    layer_id = 1,
    dynamic = True
)
class HYPER_CMS_Block(nn.Module):
    def __init__(self, config: HYPER_CMS_config):
        super().__init__()
        self.dim = config.dim
        self.depth = config.depth
        self.residual_streams = config.residual_streams
        self.rate = config.rate
        self.layer_id = config.layer_id
        self.dynamic = config.dynamic
        self.param("Static_beta", nn.ones_init(), self.rate)
        init_alpha = nn.initializers.zeros((self.dim, 1))
        init_alpha[(self.layer_id * self.dim): 1] = 1.0
        self.static_alpha = self.param("Static_Alpha", jax.numpy.concatenate([init_alpha[0], jax.numpy.eye(N=(self.rate), M=1)]))

        if self.dynamic:
            self.dynamic_alpha_fn = self.param("dynamic_alpha_fn", nn.initializers.zeros, (self.dim, self.rate + 1))
            self.dynamic_alpha_scale = self.param("dynamic_alpha_scale", lambda k, s: jnp.ones(s) * 0.01, (1,))
            self.dynamic_beta_fn = self.param("dynamic_beta_fn", nn.initializers.zeros, (self.dim,))
            self.dynamic_beta_fn_scale = self.param("dynamic_beta_scale", lambda k, s: jnp.ones(s) * 0.01, (1,))
            self.layer_norm = nnx.LayerNorm(config.dim)

    def width(self, residual):
        global norm_h
        if self.dynamic:
            norm_h = self.layer_norm(residual)


        if self.dynamic:
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = math.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]


        if self.dynamic:
            dc_weight = norm_h @ self.dynamic_beta_fn
            dc_weight = math.tanh(dc_weight)
            dynamic_beta = dc_weight @ self.dynamic_beta_fn
            beta = dynamic_beta + self.dynamic_beta[None, None, ...]
        else:
            beta = self.dynamic_beta[None, None, ...]


        mix_h = alpha.transpose(-1, -2) @ residual





    def depth(self, residual):

    def __call__(x):


class mHC_CMS_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = sche.Schedule(config)
    def __call__(x):

