import math
from dataclasses import dataclass

import jax
import optax
import flax
import optax.schedules as sche
import flax.linen as nn
import flax.nnx as nnx
import jax.numpy as jnp
from typing import Tuple, Callable, Optional
import einops
from functorch.dim import tree_flatten
from torch.utils._cxx_pytree import tree_unflatten

from titan.memory import default




# Some tools/utils

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    @nn.compact
    def __call__(self, x):
        scale = self.dim ** -0.5
        gamma = self.param("gamma", nn.initializers.zeros, (self.dim,))

        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        normalized = x / (norm + 1e-8)
        return (gamma + 1) * normalized * scale


def L2Norm(x, axis=-1):
    return jnp.linalg.norm(x, ord=1, axis=axis, keepdims=True)


def sinkhorn_knopp(log_alpha, iter: int = 20):
    """
    Sinkhorn Knopp algorithm to normalize matrix connections
    :param alpha: matrix value(s)
    :param iter:
    :return:
    """

    alpha = jnp.exp(log_alpha)

    for _ in range(iter):
        L2Norm(alpha, axis=-2)
        L2Norm(alpha, axis=-1)

    return alpha




# Vanilla Residual block

def default(v, d):
    return v if v is not None else d


def identity(x):
    return x


class Residual_CMS_Block(nn.Module):
    branch: Optional[nn.Module] = None
    residual_transformation: Optional[Callable] = None

    def setup(self):
        self._residual_transform = default(self.residual_transformation, identity)

    def width_connection(self, residuals):
        return residuals, residuals, dict()

    def depth_connection(self, branch_output, residuals, **kwargs):
        return branch_output + self._residual_transform(residuals)

    def decorate_branch(self, branch: Callable):
        assert self.branch is None, 'branch already decorated'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            return add_residual(branch_output)

        return forward_and_add_residual

    def __call__(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if self.branch is None:
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)



# hyper-connection block
class HYPER_CMS_Block(nn.Module):
    """Basic Hyper-Connections block"""

    dim: int
    depth: int
    residual_streams: int
    rate: int
    layer_id: int
    dynamic: bool = True
    branch: Optional[nn.Module] = None

    def setup(self):
        init_residual_index = self.layer_id % self.residual_streams


        init_alpha0 = jnp.zeros((self.residual_streams, 1))
        init_alpha0 = init_alpha0.at[init_residual_index, 0].set(1.0)

        self.static_alpha = self.param(
            "static_alpha",
            lambda key, shape: jnp.concatenate([init_alpha0, jnp.eye(self.residual_streams)], axis=1),
            (self.residual_streams, self.residual_streams + 1)
        )

        self.static_beta = self.param(
            "static_beta",
            nn.initializers.ones,
            (self.residual_streams,)
        )

        if self.dynamic:
            self.dynamic_alpha_fn = self.param(
                "dynamic_alpha_fn",
                nn.initializers.zeros,
                (self.dim, self.residual_streams + 1)
            )
            self.dynamic_alpha_scale = self.param(
                "dynamic_alpha_scale",
                lambda key, shape: jnp.ones(shape) * 0.01,
                (1,)
            )

            # Dynamic beta projection weights
            self.dynamic_beta_fn = self.param(
                "dynamic_beta_fn",
                nn.initializers.zeros,
                (self.dim,)
            )
            self.dynamic_beta_scale = self.param(
                "dynamic_beta_scale",
                lambda key, shape: jnp.ones(shape) * 0.01,
                (1,)
            )

            self.norm = RMSNorm(self.dim)

    def width(self, residual):
        global norm_h
        if self.dynamic:
            norm_h = self.layer_norm(residual)


        if self.dynamic:
            #Dynamic alpha
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = math.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]

            # Dynamic beta
            dc_weight = norm_h @ self.dynamic_beta_fn
            dc_weight = math.tanh(dc_weight)
            dynamic_beta = dc_weight @ self.dynamic_beta_fn
            beta = dynamic_beta + self.dynamic_beta[None, None, ...]


        else:
            alpha = self.static_alpha[None, None, ...]
            beta = self.dynamic_beta[None, None, ...]

        mix_h = jnp.einsum('...se, ...sd -> ...ed', alpha, residual)


        branch_input = mix_h[..., 0, :]  # (batch, seq, dim)
        residuals = mix_h[..., 1:, :]  # (batch, seq, streams, dim)

        return branch_input, residuals, beta

    def depth(self, branch_output, residuals, beta):
        output = jnp.einsum('...d, ...s -> ...sd', branch_output, beta)
        return output + residuals

    def __call(self, residuals, *branch_args, **branch_kwargs):

        branch_input, residual, beta = self.width(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, beta)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if self.branch is None:
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)




# manifold contained cms block
class mHC_CMS_Block(nn.Module):

    dim: int
    num_residual_steams: int
    channel_first: bool
    branch: Optional[nn.Module] = None
    layer_id: int = None
    sinkhole_iters: int = 20,
    dropout_rate: float = 0.1


    def setup(self):
        streams = self.num_residual_steams

        init_alpha = jnp.zeros((streams, 1))
        init_alpha0 = init_alpha.at[streams, 0].set(1.0)

        self.static_alpha = self.param("static_alpha",
                                       lambda key, shape: jnp.concatenate([init_alpha0, jnp.eye(streams)], axis=1),
                                       (streams, streams + 1)
                                       )

        self.static_beta = self.param("static_beta",
                                      nn.initializers.ones,
                                      (streams,))



        self.dynamic_alpha_fn = self.param(
            "dynamic_alpha_fn",
            nn.initializers.zeros,
            (self.dim, streams + 1)
        )
        self.pre_branch_scale = self.param(
            "pre_branch_scale",
            lambda key, shape: jnp.ones(shape) * 0.01,
            (1,)
        )
        self.residual_scale = self.param(
            "residual_scale",
            lambda key, shape: jnp.ones(shape) * 0.01,
            (1,)
        )

        self.dynamic_beta_fn = self.param(
            "dynamic_beta_fn",
            nn.initializers.zeros,
            (self.dim,)
        )

        self.beta_scale = self.param("beta_scale",
                                     lambda key, shape: jnp.ones(shape) * 0.01,
                                     (1,))

        self.norm = RMSNorm(self.dim)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(rate=self.dropout_rate)


    def width(self, residuals):

        streams = self.num_residual_steams

        normed = self.norm(residuals)

        wc_weight = jnp.einsum('...sd, de -> ...se', normed, self.dynamic_alpha_fn)

        # Apply separate scales for pre and residual
        alpha_scale = jnp.concatenate([
            jnp.broadcast_to(self.pre_branch_scale, (1,)),
            jnp.broadcast_to(self.residual_scale, (streams,))
        ])
        dynamic_alpha = wc_weight * alpha_scale

        alpha = dynamic_alpha + self.static_alpha

        alpha_pre = alpha[..., :1]
        alpha_res = alpha[..., 1:]

        alpha_pre = jax.nn.sigmoid(alpha_pre)
        alpha_res = sinkhorn_knopp(alpha_res, self.sinkhorn_iters)

        alpha = jnp.concatenate([alpha_pre, alpha_res], axis=-1)

        dc_weight = jnp.einsum('...sd, d -> ...s', normed, self.dynamic_beta_fn)
        dynamic_beta = dc_weight * self.beta_scale
        beta = dynamic_beta + self.static_beta
        beta = jax.nn.sigmoid(beta) * 2

        mix_h = jnp.einsum('...se, ...sd -> ...ed', alpha, residuals)

        branch_input = mix_h[..., 0, :]
        residuals = mix_h[..., 1:, :]

        return branch_input, residuals, beta


    def depth(self, branch_output, residuals, beta):

        output = jnp.einsum('...d, ...s -> ...sd', branch_output, beta)
        result = output + residuals

        if self.dropout_rate > 0:
            result = self.dropout(result)

        return result

    def decorate_branch(self, branch: Callable):
        assert self.branch is None, 'branch already wrapped on init'

        def forward_and_add_residual(residuals, *args, **kwargs):
            branch_input, add_residual = self(residuals)
            branch_output = branch(branch_input, *args, **kwargs)
            return add_residual(branch_output)

        return forward_and_add_residual

    @nn.compact
    def __call__(self, residuals, *branch_args, **branch_kwargs):

        branch_input, residuals, beta = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, beta)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if self.branch is None:
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


