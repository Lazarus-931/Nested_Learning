from typing import Sequence

import jax
import flax
import optax
import jax.numpy as jnp
import flax.linen as nn

from flax.linen import compact
from flax.nnx import Param
from jax import random

from src.assoc_memory import NestedBlock



class GeLU(nn.Module):
    def __init__(self):



class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.gamma = self.param("gamma", nn.initializers.zeros, (self.dim,))

    @nn.compact
    def __call__(self, x):
        return self.norm(x) * (self.gamma + 1)


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

class ResidualNorm(nn.Module):
    def __init__(self,
                 dim,
                 model: nn.Module,
                 out_dim = None,
                 **kwargs
                 ):
        super().__init__()
        assert out_dim is None or out_dim == dim, "out_dim and dim must match or out_dim must be None"
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.model = model
        self.out_dim = out_dim

    @nn.compact
    def __call__(self, x, pattern: str | list[str] | None = None):
        out = self.module(x, pattern=pattern)
        return self.norm(out) + x


class standardMLP(nn.Module):
    def __init__(self,
                 dim,
                 out_dim = None,
                 **kwargs
                 ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.out_dim = out_dim


    @nn.compact
    def __call__(self, x, pattern: str | list[str] | None = None):
        raise NotImplementedError


# Vanilla Memory MLP

class MemoryMLP(NestedBlock):
    def __init__(self,
                 dim,
                 layers,
                 depth,
                 activation = nn.gelu(),
                 factor: int = 4,
                 layernorm: bool = False,
                 bias: bool = False,
                 dim_out: int = None
                 ):
        super(dim).__init__(dim=dim)
        dim_hidden = (dim * factor)
        d_out = dim if dim_out is None else dim_out
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim_out)
        key = jax.random.PRNGKey(0)


        self.weights = [
            self.param(f"W_{i}", nn.initializers.normal(), (d_in, d_out))
            for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:]))
        ]

        if bias:
            self.bias = [
                self.param(f"B_{i}", nn.initializers.normal(), (d_out,))
                for i, d_out in enumerate(self.dims[1:])
            ]

        for weight in self.weights:
            jax.nn.initializers.xavier_uniform(int(weight))


    @nn.compact
    def __call__(self, *args, **kwargs):
        pass




#

class GatedResidualMemoryMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
