# https://medium.com/@bingqian/analyzing-andrej-karpathys-implementation-of-causalselfattention-in-transformers-6cc1ff41d0ee

import math

import flax
import jax.numpy as jnp
import flax.linen as nnx
from flax import nnx
import equinox as eqx



def create_causal_mask(i, j, device):
    return nnx.ones(dtype=jnp.bool_, shape=(i, j), device=device)

class CausalConfig:
    dim: int
    head_size: int
    head: int
    block_size: int

class CausalAttention(eqx.Module):
    """CausalAttention module."""
    def __init__(self, config: CausalConfig):
        super().__init__(config)
        self.dim = config.dim
        self.head_size = config.head_size
        self.head = config.head

        self.c_attn = flax.nnx.Linear(in_features=self.dim, out_features=(3 * self.dim), bias=False)

        self.c_proj = flax.nnx.Linear(in_features=self.dim, out_features=self.dim, bias=False)

        self.register_buffer(
            'bias',
            jnp.tril(jnp.ones((config.block_size, config.block_size)).reshape(1, 1, config.block_size, config.block_size)),
        )

    def __call__(self, x):
        B, T, C = x.size()


        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        att = flax.nnx.softmax(att, axis=-1)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)

        return y
