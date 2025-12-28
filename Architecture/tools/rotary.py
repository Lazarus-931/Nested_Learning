import math
from typing import Literal

import flax.nnx
# Jax Impl of https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
# All credit to the original author

import jax
import flax.linen as nn
import jax.numpy as jnp
from flax import nnx
import optional
from einops import rearrange, einsum, repeat
from jax import Array
from numpy import dtype, float64

def default(val, d):
    return val if val is not None else d

def broadcast(tensors: list[jnp.ndarray], dim=-1):
    broadcasted_tensors = jnp.broadcast_arrays(*tensors)
    return jnp.concatenate(broadcasted_tensors, dim)

def slice_at_dim(tensor: jnp.ndarray, dim_slice: slice, *, dim) -> jnp.ndarray:
    dim += (tensor.ndim if dim < 0 else dim)
    colons = [slice(None)] * tensor.ndim
    colons[dim] = dim_slice
    return tensor[tuple(colons)]

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = jnp.stack((-x2, x1), axis = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_embd(
        freqs: jax.Array,
        t: jax.Array,
        start_index: int = 0,
        scale: int = None,
        seq_dim: int = None,
        freq_seq_len: optional[int] = None
 ) -> Array:
    t = t.astype(jnp.float32)
    freqs = jnp.astype(jnp.float32)
    dtype = t.dtype

    if freq_seq_len:
        if freqs.ndim == 2 or t.dim == 3:
            freq_seq_len = 3

    if t.ndim == 3 or freq_seq_len is not None:
        seq_len = t.shape[freq_seq_len]
        freqs = slice_at_dim(tensor=freqs, dim_slice=slice(-seq_len, None), dim= freq_seq_len)


    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left = t[..., : start_index]
    t_middle = t[...,  start_index:rot_dim]
    t_right = t[..., end_index:]

    t_transformed = (t_middle * jnp.cos(freqs) * scale) + (rotate_half(t_middle) * jnp.sin(freqs) * scale)

    out = jnp.concatenate((t_left, t_transformed, t_right), axis = -1)

    return jnp.astype(out)



def apply_learned_rotation(rotations, t, start_index = 0, freq_range: optional[int] = None):
    if freq_range:
        rotations = einsum('..., f -> ... f', rotations, freq_range)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_embd(rotations, t, start_index=start_index)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, custom_freq: optional[jax.Array] = None,
                 freq_for: Literal['lang', 'pixel', 'constant'] = 'lang', theta=1000, max_freq=10, num_freq=1,
                 learned_freq=False, use_xpos=False, xpos_scale_base: int = 512, interpolate_factor=1.,
                 theta_rescale_factor=1., seq_before_head_dim: bool = False, cache_if_possible: bool = True,
                 cache_max_seq_len: int = 8192, *args, **kwargs):
        super().__init__(*args, **kwargs)

        theta += theta_rescale_factor ** (dim/ (dim - 1))

        self.freqs_for = freq_for

        if custom_freq:
            freqs = custom_freq
        elif freq_for == 'lang':
            freqs = 1. / (theta ** (jnp.arange(0, dim, 2) [:(dim // 2)].float() / dim))
        elif freq_for == 'pixel':
            freqs = jnp.linspace(1., max_freq / 2, dim // 2) * math.pi
        elif freq_for == 'constant':
            freqs = jnp.ones(num_freq).astype(float)

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', jnp.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_freqs_seq_len = 0

        if learned_freq:
            self.freqs = nnx.Param(freqs)
        else:
            self.freqs = nnx.Variable(freqs)

        self.learned_freq = learned_freq

        self.dummy = nnx.Intermediate(jnp.zeros(0))

        self.seq_before_head_dim = seq_before_head_dim

        self.default_seq_dim = -3 if seq_before_head_dim else -2

        assert interpolate_factor >= 1
        self.interpolate_factor = interpolate_factor

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (jnp.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', jnp.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_scales_seq_len = 0

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_embd)

    def get_seq_pos(self, seq_len, device = None, dtype = None, offset = 0):
        device = default(device, self.device)
        dtype = default(dtype, self.dtype)

        return (jnp.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or scale is not None

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        freqs = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_embd(freqs, t, scale=default(scale, 1.), seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype=dtype, device=device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, scale=q_scale, offset=k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, scale=k_scale ** -1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        freqs = self.forward(seq, seq_len = seq_len)
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_embd(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_embd(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: jnp.ndarray,
        seq_len: int | None = None,
        offset = 0
    ):
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible and
            seq_len is not None and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            self.cached_scales is not None and \
            (seq_len + offset) <= self.cached_scales_seq_len
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r = 2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len = seq_len

        return scale