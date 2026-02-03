# Nested Learning, from its core, is about components nested an associative memory layers. To represent this,
# one has to develop a representation of a nested layer and nested block, which the former consists of n > 1 of the latter.
from abc import abstractclassmethod
from typing import Optional, Any, Callable

import jax
import flax
import optax
import jax.numpy as jnp
from core import block_type
import flax.linen as nn
import einops


class NestedBlock(nn.Module):
    """
    Nested Block class, consisting of scaffolding to support different components of the computational sequence
    model. Optimizers, networks, etc. all are driven from this, used in NestedBlock. It represents a general
    key -> value mapping defined in definition1.

    Use cases:

    LinearAttentionBlock(NestedBlock)
    MLPBlock(NestedBlock)
    MuonOptimizerBlock(NestedBlock)

    Represents an associative memory block


    :type NestedBlock: jax.nn.Module
    """

    def __init__(self,
                 name: str,
                 chunk: int,
                 dim: int,
                 context_stream: Optional[Any],
                 objective,
                 loss_fn_outer: Callable,
                 loss_fn_inner: Callable,
                 inner_optimizer: Callable | tuple[Callable, Callable],
                 outer_optimizer: Callable | tuple[Callable, Callable],
                 ):
        super(NestedBlock, self).__init__()
        self.block_name = name
        self.chunk = chunk
        self.dim = dim
        self.type = type
        self.context_stream = context_stream
        self.objective = objective
        self.inner_optim = self.param("inner_optim", inner_optimizer)
        self.outer_optim = self.param("outer_optim", outer_optimizer)


    @classmethod
    def update(cls):
        raise NotImplementedError

    @classmethod
    def __call__(cls, x):
        raise NotImplementedError



class AssociativeMemory(NestedBlock):
    pass