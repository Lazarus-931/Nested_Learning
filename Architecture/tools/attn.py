import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass
class CausalConfig:
    dim: int
    head_size: int
    n_head: int
    block_size: int
    dropout: float
    flash: bool


class CausalAttention(nn.Module):
    config: CausalConfig

    def setup(self):
        # Remove c_attn - let caller provide q,k,v
        self.c_proj = nn.Dense(
            features=self.config.dim,
            use_bias=False
        )

    @nn.compact
    def __call__(self, q, k, v, mask=None, attn_bias=None, train: bool = False):
        """
        q, k, v: (B, n_head, T, head_size)
        mask: (T, T) or (B, n_head, T, T) - True = attend, False = mask
        attn_bias: (B, n_head, T, T) - additive bias to logits
        """
        B, n_head, T, head_size = q.shape

        if self.config.flash:
            # JAX FlashAttention
            y = nn.dot_product_attention(
                query=q,
                key=k,
                value=v,
                bias=attn_bias,  # additive bias to logits
                mask=mask,  # boolean mask
                deterministic=not train,
                dropout_rate=self.config.dropout if train else 0.0,
                dropout_rng=self.make_rng('dropout') if train else None,
                is_causal=(mask is None)  # auto-causal if no mask provided
            )
        else:
            # Standard attention
            scale = 1.0 / math.sqrt(head_size)
            att = (q @ k.transpose(0, 1, 3, 2)) * scale

            # Apply bias if provided
            if attn_bias is not None:
                att = att + attn_bias

            # Apply mask
            if mask is not None:
                att = jnp.where(mask, att, float('-inf'))
            else:
                # Default causal mask
                causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
                att = jnp.where(causal_mask, att, float('-inf'))

            att = jax.nn.softmax(att, axis=-1)

            if train and self.config.dropout > 0:
                att = nn.Dropout(rate=self.config.dropout)(att, deterministic=False)

            y = att @ v

        # Reshape: (B, n_head, T, head_size) -> (B, T, dim)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        y = self.c_proj(y)

        return y