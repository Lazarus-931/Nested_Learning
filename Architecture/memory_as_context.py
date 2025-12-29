from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import flax
import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.numpy as jnp
from flax import nnx
from functools import partial
import flax.nnx.nn.linear as fnnl

from tools.attn import CausalAttention, CausalConfig
from einops import rearrange, reduce, repeat

from tools.rotary import RotaryEmbedding

LinearNoBias = partial(fnnl.Linear, bias=False)

AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))


MACAttention_Config = CausalConfig(
    dim = 512,
    head_size = 8,
    n_head = 64,
    block_size = 1024,
    dropout = 0.1,
    flash = True
)

@dataclass
class MAC_Config:
    num_tokens: int
    dim: int
    dept: int
    segment_len: int
    neural_mem_seg_len = None,
    neural_meme_gate_attn_output = False
    neural_mem_add_value


def terp(input, end, weight):
    return ((1-weight) * input) + (weight * end)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return jnp.pad(t, ((0, 0), (0, 0), (2, 3)), constant_values=0)

def pad_seg_with_inverse(seq, seg_len, fold_into_batch: bool = True, inverse_remove_pad: bool = True):
    batch, seq_len = seq.shape[:2]
    next_seq = seq_len // (seg_len * seg_len)

    padding = next_seq - seq_len

    needs_pad = padding > 0

    if needs_pad:
        seq = jnp.pad(seq, (0, 0, 0, padding))

    def inverse(out):
        if fold_into_batch:
            out = rearrange(tensor=out, pattern='b (w n) d -> (b w) n d', n=seg_len)

        if needs_pad and inverse_remove_pad:
            out = out[...,  :-padding, :]

        return out

    return seq, inverse

 class Casual_Segmented_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, accept_residual, seg_len, num_persistent, num_longterm):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        dim_inner= heads * dim_head

        self.attn = CausalAttention.config = MACAttention_Config

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_learned_v_mix = nnx.Sequential(
            fnnl.Linear(dim, heads),
            rearrange(pattern='b n h -> b h n 1'),
            nnx.sigmoid()
        ) if accept_residual else None

        self.seg_len = seg_len

        self.split_heads = rearrange(pattern = 'b n (h d) -> b h n d', h = heads)

        self.merge_heads = lambda x: rearrange('b h n d -> b n (h d)')

        self.persistent_memory = nnx.Param(
                    nnx.initializers.normal(stddev=0.02)(
                        jax.random.PRNGKey(0),
                        (num_persistent, self.dim)
                    )
        )

        self.rotary_embd = RotaryEmbedding(dim_head)

    def __call__(self, seq: jnp.ndarray, value_residual, output_gating = False, cache: Optional = None):
        seg_len, mem_tokens = self.seg_len, self.num_longterm

        total_seg = seg_len + mem_tokens

        batch, seq_len = seq.shape[:2]

        seq, inverse_seg = pad_seg_with_inverse(seq, seg_len, fold_into_batch = False)

        initial_1 = (self.norm(seq))

        q, k, v = jnp.split(initial_1, 3, axis = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        original_v = v

        if self.to_learned_v_mix:
            mix = self.to_learned_v_mix(seq)
            v = terp(v, value_residual, mix)

        next_cache = tuple(map(inverse_seg, (k, v)))

        q, k = self.rotary_embd.rotate_queries_with_cached_keys(q, k)

        q, k, v = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = total_seg) for t in (q, k, v))

        attend_kwargs = dict()

        if self.sliding:
            k, v = tuple(rearrange(t, '(b w) ... -> b w ...', b=batch) for t in (k, v))
            k, v = tuple(pad_at_dim(t, (1, 0), value=0., dim=1) for t in (k, v))
            k = jnp.concatenate((k[:, :-1], k[:, 1:]), axis=-2)
            v = jnp.concatenate((v[:, :-1], v[:, 1:]), axis=-2)
            k, v = tuple(rearrange(t, 'b w ... -> (b w) ...') for t in (k, v))

            # take care of masking

            idx = jnp.arange(seq.shape[-2], device=seq.device)
            q_idx = rearrange(idx, '(w n) -> w n', n=seg_len)
            k_idx = pad_at_dim(q_idx, (1, 0), dim=0, value=-1e4)
            k_idx = jnp.concatenate((k_idx[:-1], k_idx[1:]), axis=-1)

            q_idx = rearrange(q_idx, 'w i -> w i 1')
            k_idx = rearrange(k_idx, 'w j -> w 1 j')

            sliding_mask = (q_idx - k_idx) <= total_seg
            sliding_mask = jnp.pad(sliding_mask, (self.num_persist_mem_tokens, 0), value=True)

            sliding_mask = repeat(sliding_mask, 'w i j -> (b w) 1 i j', b=batch)
            attend_kwargs.update(mask=sliding_mask)

            # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # persistent memory

        k = jnp.concatenate((pmk, k), axis = -2)
        v = jnp.concatenate((pmv, v), axis = -2)

        # attention

        out, _ = self.attend(q, k, v, **attend_kwargs)

        out = self.merge_heads(out)

        out_arr = jnp.asarray(out)

        out = self.to_out(out_arr)

        out = rearrange(out, '(b w) n d -> b (w n) d', b = batch)

        out = inverse_seg(out)

        if output_gating is not None:
            out = out * output_gating

        return out, AttnIntermediates(original_v, next_cache)
class MAC_Transformer(nn.Module):
    def __init__(self, config: MAC_Config):
        self.token_emb = config.token_emb
        super().__init__()



