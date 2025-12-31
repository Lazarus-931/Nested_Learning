import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass
class AttentionConfig:
    """Unified config for causal and sliding window attention"""
    dim: int
    head_size: int
    n_head: int
    block_size: int
    dropout: float = 0.0
    window_size: Optional[int] = None  # None = full causal, int = sliding window
    use_flash: bool = True
    qk_norm: bool = False  # RMSNorm on queries and keys


class CausalAndSlidingAttention(nn.Module):
    """
    Unified attention supporting both causal and sliding window patterns.

    Pattern controlled by config.window_size:
    - window_size = None: Full causal attention (token i sees all tokens 0 to i)
    - window_size = W: Sliding window (token i sees tokens max(0, i-W) to i)

    Everything else (QKV projection, scoring, retrieval) is identical.
    """
    config: AttentionConfig

    def setup(self):

        self.c_proj = nn.Dense(
            features=self.config.dim,
            use_bias=False
        )


        if self.config.qk_norm:
            self.q_norm = nn.RMSNorm()
            self.k_norm = nn.RMSNorm()

    def create_mask(self, seq_len: int) -> jnp.ndarray:
        """
        Create attention mask based on pattern.

        Returns:
            mask: (seq_len, seq_len) boolean array
                  True = can attend, False = masked out
        """
        if self.config.window_size is None:

            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        else:

            mask = jnp.zeros((seq_len, seq_len), dtype=jnp.bool_)

            for i in range(seq_len):

                start = max(0, i - self.config.window_size + 1)
                mask = mask.at[i, start:i + 1].set(True)

        return mask

    @nn.compact
    def __call__(
            self,
            q: jnp.ndarray,
            k: jnp.ndarray,
            v: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
            attn_bias: Optional[jnp.ndarray] = None,
            train: bool = False
    ):
        """
        Standard multi-head attention forward pass.

        Args:
            q, k, v: (B, n_head, T, head_size) - pre-projected queries/keys/values
            mask: (T, T) or (B, n_head, T, T) - True = attend, False = mask
                  If None, uses pattern from config (causal or sliding)
            attn_bias: (B, n_head, T, T) - additive bias to attention logits
            train: Whether in training mode (for dropout)

        Returns:
            output: (B, T, dim) - attention output
        """
        B, n_head, T, head_size = q.shape

        # Optional QK normalization
        if self.config.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.config.use_flash:

            if mask is None:
                mask = self.create_mask(T)

            y = nn.dot_product_attention(
                query=q,
                key=k,
                value=v,
                bias=attn_bias,
                mask=mask,
                deterministic=not train,
                dropout_rate=self.config.dropout if train else 0.0,
                dropout_rng=self.make_rng('dropout') if train else None,
            )
        else:
            scale = 1.0 / math.sqrt(head_size)

            att = (q @ k.transpose(0, 1, 3, 2)) * scale

            if attn_bias is not None:
                att = att + attn_bias

            if mask is not None:
                att = jnp.where(mask, att, float('-inf'))
            else:
                pattern_mask = self.create_mask(T)
                att = jnp.where(pattern_mask, att, float('-inf'))


            att = jax.nn.softmax(att, axis=-1)


            if train and self.config.dropout > 0:
                att = nn.Dropout(rate=self.config.dropout)(
                    att,
                    deterministic=False
                )


            y = att @ v  # (B, n_head, T, head_size)


        y = y.transpose(0, 2, 1, 3).reshape(B, T, self.config.dim)


        y = self.c_proj(y)

        return y

