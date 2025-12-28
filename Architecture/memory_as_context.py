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


def terp(input, end, weight):
    return ((1-weight) * input) + (weight * end)


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

        self.attn = CausalAttention.__init__(CausalConfig)

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_learned_v_mix = nnx.Sequential(
            fnnl.Linear(dim, heads),
            rearrange(pattern='b n h -> b h n 1'),
            nnx.sigmoid()
        ) if accept_residual else None

        self.seg_len = seg_len

        self.split_heads = rearrange(pattern = 'b n (h d) -> b h n d', h = heads)

        self.merge_heads = rearrange('b h n d -> b n (h d)')

        self.persistent_memory = nnx.Param(
                    nnx.initializers.normal(stddev=0.02)(
                        jax.random.PRNGKey(0),
                        (num_persistent, self.dim)
                    )
        )

        self.rotary_embd = RotaryEmbedding(dim_head)

        self.attend = Attend

    def __call__(self, seq: jnp.ndarray, value_residual, output_gating = False, cache: Optional = None):
        seg_len, mem_tokens = self.seg_len, self.num_longterm

        total_seg = seg_len + mem_tokens

        batch, seq_len = seq.shape[:2]

        seq, inverse_seg = pad_seg_with_inverse(seq, seg_len, fold_into_batch = False)

        initial_1 = (self.norm(seq))

        q, k, v = jnp.split(initial_1, 3, axis = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        originial_v = v

        if self.to_learned_v_mix:
            mix = self.to_learned_v_mix(seq)
            v = terp(v, value_residual, mix)

        next_cache = tuple(map(inverse_seg, (k, v)))

        q, k = self.r

class MAC_Transformer(nn.Module):
    def __init__(self):
