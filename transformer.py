import flax
import jax.numpy as jnp
import numpy as np
import jax.nn as jnn
import jax.lax as lax

from memory import MemoryLayerArgs, MemoryMLP, TitanMemory

from flax import nnx


from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    transformer_layers: int
    memory_layers: int
    num_heads: int
    in_val: int
    qkv_val: int
    out_val: int
    model_layers: int
    dims: int


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024

    productkey_args: ProductKeyArgs = field(default_factory=ProductKeyArgs)


class Transformer(nnx.Module):
    def __init__(self, dim: int, config: ModelConfig, layers: int):
        super().__init__()
        self.config = config
        for i in range(layers):
            self.attention += nnx.MultiHeadAttention(
            num_heads=self.config.num_heads,
            dims=self.config.dims,
            in_features=self.config.in_val,
            qkv_features=self.config.qkv_val,
            out_features=self.config.out_val,
        )
        self.memory = TitanMemory(MemoryLayerArgs)
        self.norm = nnx.LayerNorm(dim)

    def __call__(self, x: jnp.ndarray, teach_signal=None):
        attn_output = self.attention(x)
        x += attn_output

        x = self.norm(x)

        return x
































